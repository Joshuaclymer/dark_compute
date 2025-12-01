"""
Predict AI takeoff trajectories and capability milestones from arbitrary compute time series.

This module provides a simple interface to run the progress model on custom compute
trajectories and extract milestone predictions for key AI capabilities.

Example usage:
    ```python
    import numpy as np
    from predict_trajectory import predict_milestones_from_compute

    # Define custom compute trajectory
    years = np.linspace(2024, 2050, 100)
    inference_compute = 1e6 * np.exp(0.5 * (years - 2024))  # H100-years
    experiment_compute = 1e5 * np.exp(0.4 * (years - 2024))

    # Get milestone predictions
    milestones = predict_milestones_from_compute(
        time=years,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute
    )

    # Print results
    for name, info in sorted(milestones.items(), key=lambda x: x[1]['time']):
        print(f"{name}: {info['time']:.1f} years")
    ```
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging

from progress_model import ProgressModel, TimeSeriesData, Parameters
import model_config as cfg

logger = logging.getLogger(__name__)

# Try to import compute allocation utilities
try:
    from compute_allocation import compute_allocation_heuristic
    HAS_ALLOCATION = True
except ImportError:
    HAS_ALLOCATION = False
    logger.warning("compute_allocation module not available - total_compute feature disabled")


@dataclass
class MilestoneInfo:
    """Information about a capability milestone."""
    time: float  # When milestone is achieved (decimal years)
    progress_level: float  # Progress in OOMs of effective compute
    research_effort: float  # Research effort at milestone
    research_stock: float  # Cumulative research stock at milestone
    progress_multiplier: float  # AI R&D speedup multiplier
    metric_name: str  # Which metric crossed threshold
    target_value: float  # Threshold value that was crossed


class TrajectoryPredictor:
    """Predict AI takeoff trajectories from compute time series."""

    def __init__(self, params: Optional[Parameters] = None):
        """
        Initialize trajectory predictor.

        Args:
            params: Model parameters. If None, uses default parameters.
        """
        self.params = params if params is not None else Parameters()
        self.model = None
        self.results = None

    def predict_from_time_series(
        self,
        time: np.ndarray,
        inference_compute: np.ndarray,
        experiment_compute: np.ndarray,
        L_HUMAN: Optional[np.ndarray] = None,
        training_compute_growth_rate: Optional[np.ndarray] = None,
        initial_progress: float = 0.0,
        # Weight stealing parameters - resources for leading project
        inference_compute_leading_project: Optional[np.ndarray] = None,
        experiment_compute_leading_project: Optional[np.ndarray] = None,
        L_HUMAN_leading_project: Optional[np.ndarray] = None,
        years_weights_are_stolen_from_leading_project: Optional[List[Union[float, str]]] = None,
        stealing_algorithms_up_to: Optional[Union[float, str]] = None,
        # Capability cap - upper bound on progress at each time step
        capability_cap: Optional[np.ndarray] = None
    ) -> Dict[str, MilestoneInfo]:
        """
        Predict capability milestones from compute time series.

        Args:
            time: Time points (decimal years), e.g. [2024.0, 2024.5, 2025.0, ...]
            inference_compute: AI inference compute at each time (H100-years)
            experiment_compute: Experiment compute budget at each time (H100-years)
            L_HUMAN: Human labor supply at each time (optional, defaults to constant)
            training_compute_growth_rate: Training compute growth rate (OOMs/year, optional)
            initial_progress: Starting progress level in OOMs (default: 0.0)
            inference_compute_leading_project: AI inference compute for leading project (optional)
            experiment_compute_leading_project: Experiment compute for leading project (optional)
            L_HUMAN_leading_project: Human labor for leading project (optional)
            years_weights_are_stolen_from_leading_project: List of years or milestone names when
                weights are stolen. Can contain floats (decimal years, e.g., 2030.5) or strings
                (milestone names, e.g., "SC", "AC", "ASI"). Milestone names are resolved from the
                leading project's trajectory. (optional)
            stealing_algorithms_up_to: Time or milestone up to which the stealing project has the
                same software efficiency as the leading project. Can be a float (decimal year) or
                string (milestone name). This models the scenario where algorithms/techniques are
                stolen from the leading project. (optional)
            capability_cap: Optional time series of progress caps. At each time step, progress will
                be upper bounded by the corresponding value in this array. This models scenarios where
                capabilities are externally constrained (e.g., by regulatory caps or evaluation-based
                limits). Must have the same length as the time array if provided. (optional)

        Returns:
            Dictionary mapping milestone names to MilestoneInfo objects.
            Milestones include:
                - AC: Automated Coder
                - AI2027-SC: Superhuman Coder
                - AIR-5x, AIR-25x, AIR-250x, AIR-2000x, AIR-10000x: AI R&D multipliers
                - SAR: Superhuman AI Researcher
                - SIAR: Significantly Improved AI Researcher
                - STRAT-AI: Strategic AI
                - TED-AI: Transformative Economic Disruption AI
                - ASI: Artificial Superintelligence

        Weight Stealing Mode:
            When inference_compute_leading_project, experiment_compute_leading_project, or
            L_HUMAN_leading_project are provided along with years_weights_are_stolen_from_leading_project,
            the model simulates a scenario where:
            1. A "leading project" progresses with its own resources (the _leading_project arrays)
            2. The "stealing project" progresses with the base resources (inference_compute, etc.)
            3. At each stealing event, the stealing project gains access to the leading project's
               AI CAPABILITIES (automation fraction, AI research taste) but keeps its own progress
               and research stock. This accelerates future progress by providing a more capable AI.

        Algorithm Stealing Mode:
            When stealing_algorithms_up_to is provided, the stealing project has the same SOFTWARE
            EFFICIENCY as the leading project up to the specified time/milestone. This models
            scenarios where algorithms and techniques are stolen, giving the stealing project
            the benefit of the leading project's software improvements without stealing the
            model weights themselves.

            Weight stealing and algorithm stealing can be combined.
        """
        # Validate inputs
        if not (len(time) == len(inference_compute) == len(experiment_compute)):
            raise ValueError("time, inference_compute, and experiment_compute must have same length")

        # Set defaults for optional inputs
        if L_HUMAN is None:
            # Default to 100 software engineers (reasonable present-day baseline)
            L_HUMAN = np.ones_like(time) * 100.0

        if training_compute_growth_rate is None:
            # Default to 0.5 OOMs/year (moderate training compute growth)
            training_compute_growth_rate = np.ones_like(time) * 0.5

        # Validate capability_cap if provided
        if capability_cap is not None and len(capability_cap) != len(time):
            raise ValueError("capability_cap must have same length as time array")

        # Check if we're in weight stealing mode or algorithm stealing mode
        has_weight_stealing = (
            years_weights_are_stolen_from_leading_project is not None and
            len(years_weights_are_stolen_from_leading_project) > 0
        )
        has_algorithm_stealing = stealing_algorithms_up_to is not None
        weight_stealing_mode = has_weight_stealing or has_algorithm_stealing

        # Make a copy of params to avoid modifying the original
        import copy
        working_params = copy.deepcopy(self.params)

        if weight_stealing_mode:
            # Set up leading project resources - default to same as base if not provided
            if inference_compute_leading_project is None:
                inference_compute_leading_project = inference_compute.copy()
            if experiment_compute_leading_project is None:
                experiment_compute_leading_project = experiment_compute.copy()
            if L_HUMAN_leading_project is None:
                L_HUMAN_leading_project = L_HUMAN.copy()

            # Validate leading project inputs
            if not (len(time) == len(inference_compute_leading_project) ==
                    len(experiment_compute_leading_project) == len(L_HUMAN_leading_project)):
                raise ValueError("Leading project arrays must have same length as time array")

            # Update params copy for weight stealing mode
            working_params.weight_stealing_mode = True
            if has_weight_stealing:
                working_params.years_weights_are_stolen = list(years_weights_are_stolen_from_leading_project)
            if has_algorithm_stealing:
                working_params.stealing_algorithms_up_to = stealing_algorithms_up_to

        # Create time series data for the stealing project (or main project if not in weight stealing mode)
        time_series = TimeSeriesData(
            time=time,
            L_HUMAN=L_HUMAN,
            inference_compute=inference_compute,
            experiment_compute=experiment_compute,
            training_compute_growth_rate=training_compute_growth_rate,
            capability_cap=capability_cap
        )

        if weight_stealing_mode:
            # First, compute the leading project trajectory
            leading_time_series = TimeSeriesData(
                time=time,
                L_HUMAN=L_HUMAN_leading_project,
                inference_compute=inference_compute_leading_project,
                experiment_compute=experiment_compute_leading_project,
                training_compute_growth_rate=training_compute_growth_rate
            )

            # Create leading project model with same params but separate data
            from progress_model import WeightStealingProgressModel
            self.model = WeightStealingProgressModel(
                working_params,
                time_series,  # Stealing project resources
                leading_time_series  # Leading project resources
            )
        else:
            # Standard mode - just use ProgressModel
            self.model = ProgressModel(working_params, time_series)

        try:
            self.model.compute_progress_trajectory(
                time_range=[time[0], time[-1]],
                initial_progress=initial_progress
            )
            self.results = self.model.results
        except Exception as e:
            logger.error(f"Failed to compute trajectory: {e}")
            raise

        # Extract milestone information
        milestones = {}
        if 'milestones' in self.results:
            for name, info in self.results['milestones'].items():
                if 'time' in info:  # Only include achieved milestones
                    milestones[name] = MilestoneInfo(
                        time=info['time'],
                        progress_level=info.get('effective_compute_ooms', np.nan),
                        research_effort=info.get('research_effort', np.nan),
                        research_stock=info.get('research_stock', np.nan),
                        progress_multiplier=info.get('progress_multiplier', np.nan),
                        metric_name=info.get('metric', ''),
                        target_value=info.get('target', np.nan)
                    )

        return milestones

    def get_full_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Get complete trajectory after prediction.

        Returns:
            Dictionary with trajectory arrays including:
                - times: Time points
                - progress: Cumulative progress (OOMs)
                - progress_rates: Progress rate (OOMs/year)
                - automation_fraction: Fraction of SWE tasks automated
                - ai_research_taste_sd: AI experiment selection skill (std devs)
                - ai_coding_labor_mult_ref_present_day: AI coding labor multiplier
                - ai_sw_progress_mult_ref_present_day: AI R&D acceleration
                ... and many more metrics

        Raises:
            RuntimeError: If predict_from_time_series() hasn't been called yet
        """
        if self.results is None:
            raise RuntimeError("Must call predict_from_time_series() first")

        return self.results

    def get_milestone_times(self) -> Dict[str, float]:
        """
        Get just the milestone achievement times.

        Returns:
            Dictionary mapping milestone names to achievement times (decimal years).
            Returns None for milestones not achieved in the time range.
        """
        if self.results is None or 'milestones' not in self.results:
            raise RuntimeError("Must call predict_from_time_series() first")

        return {
            name: info['time']
            for name, info in self.results['milestones'].items()
        }


def predict_milestones_from_total_compute(
    time: Union[np.ndarray, List[float]],
    total_compute: Union[np.ndarray, List[float]],
    L_HUMAN: Optional[Union[np.ndarray, List[float]]] = None,
    training_compute_growth_rate: Optional[Union[np.ndarray, List[float]]] = None,
    initial_progress: float = 0.0,
    params: Optional[Parameters] = None,
    inference_fraction: float = 0.15
) -> Dict[str, MilestoneInfo]:
    """
    Convenience function to predict milestones from TOTAL compute (automatically split).

    This function automatically allocates total compute between inference and experiment
    using a simple heuristic (default: 15% inference, 85% experiment).

    Args:
        time: Time points (decimal years), e.g. [2024.0, 2025.0, 2026.0, ...]
        total_compute: Total compute budget at each time (H100-years)
        L_HUMAN: Human labor supply at each time (optional, defaults to constant)
        training_compute_growth_rate: Training compute growth rate (OOMs/year, optional)
        initial_progress: Starting progress level in OOMs (default: 0.0)
        params: Model parameters (optional, uses defaults if None)
        inference_fraction: Fraction of total compute for inference (default: 0.15)

    Returns:
        Dictionary mapping milestone names to MilestoneInfo objects.

    Example:
        ```python
        years = np.linspace(2024, 2050, 100)
        total = 1e6 * np.exp(0.5 * (years - 2024))  # Total compute

        milestones = predict_milestones_from_total_compute(
            time=years,
            total_compute=total
        )
        ```
    """
    if not HAS_ALLOCATION:
        logger.warning("Using simple fixed split - install compute_allocation for adaptive splitting")

    # Convert to arrays
    time = np.asarray(time, dtype=float)
    total_compute = np.asarray(total_compute, dtype=float)

    # Split compute using heuristic
    inference_compute = inference_fraction * total_compute
    experiment_compute = (1 - inference_fraction) * total_compute

    # Call standard prediction function
    return predict_milestones_from_compute(
        time=time,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute,
        L_HUMAN=L_HUMAN,
        training_compute_growth_rate=training_compute_growth_rate,
        initial_progress=initial_progress,
        params=params
    )


def predict_milestones_from_compute(
    time: Union[np.ndarray, List[float]],
    inference_compute: Union[np.ndarray, List[float]],
    experiment_compute: Union[np.ndarray, List[float]],
    L_HUMAN: Optional[Union[np.ndarray, List[float]]] = None,
    training_compute_growth_rate: Optional[Union[np.ndarray, List[float]]] = None,
    initial_progress: float = 0.0,
    params: Optional[Parameters] = None,
    # Weight stealing parameters
    inference_compute_leading_project: Optional[Union[np.ndarray, List[float]]] = None,
    experiment_compute_leading_project: Optional[Union[np.ndarray, List[float]]] = None,
    L_HUMAN_leading_project: Optional[Union[np.ndarray, List[float]]] = None,
    years_weights_are_stolen_from_leading_project: Optional[List[Union[float, str]]] = None,
    stealing_algorithms_up_to: Optional[Union[float, str]] = None,
    # Capability cap - upper bound on progress at each time step
    capability_cap: Optional[Union[np.ndarray, List[float]]] = None
) -> Dict[str, MilestoneInfo]:
    """
    Convenience function to predict milestones from compute time series.

    Args:
        time: Time points (decimal years), e.g. [2024.0, 2025.0, 2026.0, ...]
        inference_compute: AI inference compute at each time (H100-years)
        experiment_compute: Experiment compute budget at each time (H100-years)
        L_HUMAN: Human labor supply at each time (optional, defaults to constant)
        training_compute_growth_rate: Training compute growth rate (OOMs/year, optional)
        initial_progress: Starting progress level in OOMs (default: 0.0)
        params: Model parameters (optional, uses defaults if None)
        inference_compute_leading_project: Inference compute for leading project (optional)
        experiment_compute_leading_project: Experiment compute for leading project (optional)
        L_HUMAN_leading_project: Human labor for leading project (optional)
        years_weights_are_stolen_from_leading_project: List of years or milestone names when
            weights are stolen. Can contain floats (decimal years) or strings (milestone names
            like "SC", "AC", "ASI"). Milestone names are resolved from the leading project's
            trajectory. (optional)
        stealing_algorithms_up_to: Time or milestone up to which the stealing project has the
            same software efficiency as the leading project. Can be a float (decimal year) or
            string (milestone name like "SC"). (optional)
        capability_cap: Optional time series of progress caps. At each time step, progress will
            be upper bounded by the corresponding value in this array. (optional)

    Returns:
        Dictionary mapping milestone names to MilestoneInfo objects.

    Example (standard):
        ```python
        years = np.linspace(2024, 2050, 100)
        inference = 1e6 * np.exp(0.5 * (years - 2024))
        experiment = 1e5 * np.exp(0.4 * (years - 2024))

        milestones = predict_milestones_from_compute(
            time=years,
            inference_compute=inference,
            experiment_compute=experiment
        )
        ```

    Example (algorithm stealing up to SC milestone):
        ```python
        years = np.linspace(2024, 2050, 100)
        # Stealing project has less resources
        inference = 1e5 * np.exp(0.4 * (years - 2024))
        experiment = 1e4 * np.exp(0.3 * (years - 2024))
        # Leading project has more resources
        inference_leader = 1e6 * np.exp(0.5 * (years - 2024))
        experiment_leader = 1e5 * np.exp(0.4 * (years - 2024))

        # Steal algorithms (software efficiency) up to SC milestone
        milestones = predict_milestones_from_compute(
            time=years,
            inference_compute=inference,
            experiment_compute=experiment,
            inference_compute_leading_project=inference_leader,
            experiment_compute_leading_project=experiment_leader,
            stealing_algorithms_up_to="SC"
        )
        ```
    """
    # Convert lists to arrays
    time = np.asarray(time, dtype=float)
    inference_compute = np.asarray(inference_compute, dtype=float)
    experiment_compute = np.asarray(experiment_compute, dtype=float)

    if L_HUMAN is not None:
        L_HUMAN = np.asarray(L_HUMAN, dtype=float)
    if training_compute_growth_rate is not None:
        training_compute_growth_rate = np.asarray(training_compute_growth_rate, dtype=float)

    # Convert leading project arrays if provided
    if inference_compute_leading_project is not None:
        inference_compute_leading_project = np.asarray(inference_compute_leading_project, dtype=float)
    if experiment_compute_leading_project is not None:
        experiment_compute_leading_project = np.asarray(experiment_compute_leading_project, dtype=float)
    if L_HUMAN_leading_project is not None:
        L_HUMAN_leading_project = np.asarray(L_HUMAN_leading_project, dtype=float)

    # Convert capability_cap if provided
    if capability_cap is not None:
        capability_cap = np.asarray(capability_cap, dtype=float)

    # Create predictor and run
    predictor = TrajectoryPredictor(params=params)
    milestones = predictor.predict_from_time_series(
        time=time,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute,
        L_HUMAN=L_HUMAN,
        training_compute_growth_rate=training_compute_growth_rate,
        initial_progress=initial_progress,
        inference_compute_leading_project=inference_compute_leading_project,
        experiment_compute_leading_project=experiment_compute_leading_project,
        L_HUMAN_leading_project=L_HUMAN_leading_project,
        years_weights_are_stolen_from_leading_project=years_weights_are_stolen_from_leading_project,
        stealing_algorithms_up_to=stealing_algorithms_up_to,
        capability_cap=capability_cap
    )

    return milestones


def predict_trajectory_from_csv(
    csv_path: str,
    params: Optional[Parameters] = None,
    initial_progress: float = 0.0
) -> Tuple[Dict[str, MilestoneInfo], Dict[str, np.ndarray]]:
    """
    Predict trajectory from CSV file with compute time series.

    Args:
        csv_path: Path to CSV file with columns: time, L_HUMAN, inference_compute,
                 experiment_compute, training_compute_growth_rate
        params: Model parameters (optional, uses defaults if None)
        initial_progress: Starting progress level in OOMs (default: 0.0)

    Returns:
        Tuple of (milestones, full_trajectory)
        - milestones: Dictionary mapping milestone names to MilestoneInfo
        - full_trajectory: Dictionary with all trajectory arrays
    """
    from progress_model import load_time_series_data

    # Load time series
    time_series = load_time_series_data(csv_path)

    # Create predictor and run
    predictor = TrajectoryPredictor(params=params)
    milestones = predictor.predict_from_time_series(
        time=time_series.time,
        inference_compute=time_series.inference_compute,
        experiment_compute=time_series.experiment_compute,
        L_HUMAN=time_series.L_HUMAN,
        training_compute_growth_rate=time_series.training_compute_growth_rate,
        initial_progress=initial_progress
    )

    full_trajectory = predictor.get_full_trajectory()

    return milestones, full_trajectory


def print_milestone_summary(milestones: Dict[str, MilestoneInfo]) -> None:
    """
    Print a formatted summary of milestone predictions.

    Args:
        milestones: Dictionary of milestone predictions from predict_milestones_from_compute
    """
    print("\n" + "="*80)
    print("AI CAPABILITY MILESTONE PREDICTIONS")
    print("="*80)

    if not milestones:
        print("No milestones achieved in the specified time range.")
        return

    # Sort by time
    sorted_milestones = sorted(milestones.items(), key=lambda x: x[1].time)

    print(f"\n{'Milestone':<15} {'Year':<8} {'Progress':<12} {'AI R&D Mult':<12}")
    print("-" * 80)

    for name, info in sorted_milestones:
        print(f"{name:<15} {info.time:>7.2f}  {info.progress_level:>10.2f}  {info.progress_multiplier:>11.1f}x")

    print("\n" + "="*80)

    # Print milestone descriptions
    print("\nMilestone Descriptions:")
    print("-" * 80)
    descriptions = {
        'AC': 'Automated Coder - AI can automate software engineering tasks',
        'AI2027-SC': 'Superhuman Coder - AI coding exceeds human parallel capability',
        'AIR-5x': 'AI R&D 5x - AI accelerates software R&D by 5x',
        'AIR-25x': 'AI R&D 25x - AI accelerates software R&D by 25x',
        'AIR-250x': 'AI R&D 250x - AI accelerates software R&D by 250x',
        'AIR-2000x': 'AI R&D 2000x - AI accelerates software R&D by 2000x',
        'AIR-10000x': 'AI R&D 10000x - AI accelerates software R&D by 10000x',
        'SAR': 'Superhuman AI Researcher - Top 0.1% experiment selection skill',
        'SIAR': 'Significantly Improved AI Researcher - 3x SAR level',
        'STRAT-AI': 'Strategic AI - Strategically aware AI system',
        'TED-AI': 'Transformative Economic Disruption AI - Economy-transforming AI',
        'ASI': 'Artificial Superintelligence - Far beyond human capability'
    }

    for name, info in sorted_milestones:
        if name in descriptions:
            print(f"  {name}: {descriptions[name]}")

    print("="*80 + "\n")


if __name__ == "__main__":
    # Example: Predict milestones for exponential compute growth
    print("Example: Predicting milestones for exponential compute growth")
    print("-" * 80)

    # Define compute trajectory
    years = np.linspace(2024, 2045, 200)

    # Exponential growth: doubling every 2 years
    inference_compute = 1e6 * 2**((years - 2024) / 2.0)  # H100-years
    experiment_compute = 1e5 * 2**((years - 2024) / 2.0)  # H100-years

    print(f"Time range: {years[0]:.1f} to {years[-1]:.1f}")
    print(f"Inference compute: {inference_compute[0]:.2e} to {inference_compute[-1]:.2e} H100-years")
    print(f"Experiment compute: {experiment_compute[0]:.2e} to {experiment_compute[-1]:.2e} H100-years")

    # Predict milestones
    milestones = predict_milestones_from_compute(
        time=years,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute
    )

    # Print results
    print_milestone_summary(milestones)

    # Example: Load from CSV
    print("\nTo predict from CSV file:")
    print("  milestones, trajectory = predict_trajectory_from_csv('input_data.csv')")
