"""
US Project classes for modeling capabilities during AI slowdown.

This module provides:
1. ProxyProject: Computes capabilities trajectory for a project with compute capped
   based on PRC covert compute estimates. This serves as a proxy for what uncooperative
   actors might achieve covertly.

2. LargestUSProject: Computes the capability cap for the largest US AI company during
   an AI slowdown, based on the ProxyProject capabilities and evaluation-based limits.

See slowdown_model.md for detailed documentation of the capability cap logic.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from backend.paramaters import ProxyProjectParameters, USProjectParameters


def filter_trajectory_by_start_year(
    trajectory_data: Optional[Dict[str, Any]],
    start_year: float
) -> Optional[Dict[str, Any]]:
    """Filter trajectory data to only include points at or after the start year.

    This is used to filter proxy project and capability cap data to only show
    data from the agreement start year onwards (since these concepts don't
    exist before the agreement).

    Args:
        trajectory_data: Dict with 'trajectory_times' and 'speedup_percentiles'
            (in MC-like format), or None
        start_year: The year to start from (points before this are removed)

    Returns:
        Filtered trajectory data dict, or None if input is None
    """
    if trajectory_data is None:
        return None

    times = trajectory_data.get('trajectory_times', [])
    if not times:
        return trajectory_data

    # Find indices where year >= start_year
    valid_indices = [i for i, t in enumerate(times) if t >= start_year]
    if not valid_indices:
        return trajectory_data

    # Filter trajectory times
    filtered_times = [times[i] for i in valid_indices]

    # Filter speedup percentiles
    speedup_percentiles = trajectory_data.get('speedup_percentiles', {})
    filtered_percentiles = {}
    for key, values in speedup_percentiles.items():
        if values and len(values) == len(times):
            filtered_percentiles[key] = [values[i] for i in valid_indices]
        else:
            filtered_percentiles[key] = values

    return {
        'trajectory_times': filtered_times,
        'speedup_percentiles': filtered_percentiles,
        'milestones_median': trajectory_data.get('milestones_median', {}),
        'asi_times': trajectory_data.get('asi_times', {})
    }


class ProxyProject:
    """
    Computes the capabilities trajectory for a proxy project.

    The proxy project represents what the PRC or other uncooperative actors might
    achieve covertly. Its compute is capped based on a percentile of the PRC
    operational covert compute distribution.

    The compute cap is updated at a specified frequency (creating step changes),
    and the capabilities trajectory is computed by running the takeoff model with
    this capped compute.
    """

    def __init__(self, params: Optional[ProxyProjectParameters] = None):
        """
        Initialize the ProxyProject.

        Args:
            params: ProxyProjectParameters containing:
                - compute_cap_as_percentile_of_PRC_operational_covert_compute: float (0-1)
                - frequency_cap_is_updated_in_years: float
        """
        self.params = params if params is not None else ProxyProjectParameters()
        self._compute_cap: Optional[Dict[str, Any]] = None
        self._trajectory: Optional[Dict[str, Any]] = None

    def compute_compute_cap(
        self,
        years: List[float],
        covert_compute_percentiles: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Compute the compute cap for the proxy project over time.

        The compute cap is determined by taking a given percentile of the PRC operational
        covert compute distribution. The cap is updated at a specified frequency,
        creating a discrete/step-change curve.

        Args:
            years: List of years for the time series
            covert_compute_percentiles: Dict with keys like 'p10', 'p25', 'p50', 'p75', 'p90'
                containing lists of compute values at each percentile for each year

        Returns:
            Dictionary with:
                - 'years': List of years
                - 'compute': List of compute cap values
        """
        if not years or not covert_compute_percentiles:
            self._compute_cap = {'years': [], 'compute': []}
            return self._compute_cap

        percentile = self.params.compute_cap_as_percentile_of_PRC_operational_covert_compute
        update_frequency = self.params.frequency_cap_is_updated_in_years

        # Determine which percentile key to use based on the percentile value
        percentile_keys = {
            0.10: 'p10',
            0.25: 'p25',
            0.50: 'p50',
            0.75: 'p75',
            0.90: 'p90'
        }

        # Find the closest available percentile
        available_percentiles = sorted(percentile_keys.keys())
        closest_percentile = min(available_percentiles, key=lambda x: abs(x - percentile))
        percentile_key = percentile_keys[closest_percentile]

        # Get the covert compute values at the selected percentile
        if percentile_key not in covert_compute_percentiles:
            # Fallback to median if the exact percentile isn't available
            percentile_key = 'p50' if 'p50' in covert_compute_percentiles else 'median'

        if percentile_key not in covert_compute_percentiles:
            self._compute_cap = {'years': [], 'compute': []}
            return self._compute_cap

        covert_compute_at_percentile = covert_compute_percentiles[percentile_key]

        # Build the step-change curve
        cap_values = []
        current_cap = None
        last_update_year = None

        for i, year in enumerate(years):
            if last_update_year is None or (year - last_update_year) >= update_frequency:
                # Update the cap
                current_cap = covert_compute_at_percentile[i]
                last_update_year = year

            cap_values.append(current_cap if current_cap is not None else 0)

        self._compute_cap = {
            'years': list(years),
            'compute': cap_values
        }
        return self._compute_cap

    def compute_trajectory(
        self,
        covert_compute_data: Optional[Dict[str, Any]],
        takeoff_model: Any = None,
        human_labor: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute the full capabilities trajectory for the proxy project.

        This runs the takeoff model with the compute-capped inputs to get
        the AI R&D speedup trajectory.

        Args:
            covert_compute_data: Dict with 'years' and 'operational_dark_compute' containing
                percentile data (p25, median/p50, p75)
            takeoff_model: TakeoffModel instance for computing trajectories
            human_labor: Optional list of human labor values. If None, uses default.

        Returns:
            Dictionary with trajectory data including 'speedup_percentiles' or None if
            insufficient data
        """
        if not covert_compute_data:
            return None

        years = covert_compute_data.get('years', [])
        operational = covert_compute_data.get('operational_dark_compute', {})

        # Build percentile dict from operational dark compute data
        covert_compute_percentiles = {}
        if 'p25' in operational:
            covert_compute_percentiles['p25'] = operational['p25']
        if 'median' in operational:
            covert_compute_percentiles['p50'] = operational['median']
        if 'p75' in operational:
            covert_compute_percentiles['p75'] = operational['p75']

        if not years or not covert_compute_percentiles:
            return None

        # Compute the compute cap
        compute_cap_data = self.compute_compute_cap(years, covert_compute_percentiles)

        if not compute_cap_data['compute']:
            return None

        # If we have a takeoff model, run trajectory prediction
        if takeoff_model is not None:
            if human_labor is None:
                # Default human labor - could be parameterized
                human_labor = [100.0] * len(years)

            self._trajectory = takeoff_model.predict_trajectory_deterministic(
                time=years,
                human_labor=human_labor,
                compute=compute_cap_data['compute']
            )
            if self._trajectory:
                self._trajectory['compute_cap'] = compute_cap_data
        else:
            # No takeoff model - just store compute cap
            self._trajectory = {'compute_cap': compute_cap_data}

        return self._trajectory

    @property
    def compute_cap(self) -> Optional[Dict[str, Any]]:
        """Get the computed compute cap."""
        return self._compute_cap

    @property
    def trajectory(self) -> Optional[Dict[str, Any]]:
        """Get the computed trajectory."""
        return self._trajectory

    def get_speedup_at_time(self, year: float) -> Optional[float]:
        """
        Get the AI R&D speedup at a specific time.

        Args:
            year: The year to query

        Returns:
            The AI R&D speedup at that time, or None if trajectory not computed
        """
        if self._trajectory is None or 'speedup' not in self._trajectory:
            return None

        years = self._trajectory.get('trajectory_times', [])
        speedups = self._trajectory.get('speedup', [])

        if not years or not speedups:
            return None

        # Interpolate to find speedup at the requested year
        return float(np.interp(year, years, speedups))


class LargestUSProject:
    """
    Computes the capability cap and trajectory for the largest US AI company during an AI slowdown.

    The capability cap bounds the AI R&D speedup that US projects can achieve.

    Before evaluation-based caps are implemented:
        capability_cap = proxy_project_capabilities

    After evaluation-based caps are implemented:
        capability_cap = max(fixed_cap, proxy_project_capabilities)

    where fixed_cap is the capability_cap_ai_randd_speedup parameter (by default,
    set to the handoff threshold where developers cannot effectively oversee alignment).

    The US project trajectory is computed by running the takeoff model with the
    capability cap passed in, which upper-bounds progress at each time step.

    See slowdown_model.md section "Capability cap" and "US project capabilities during
    the AI slowdown" for full documentation.
    """

    def __init__(self, params: Optional[USProjectParameters] = None):
        """
        Initialize the LargestUSProject.

        Args:
            params: USProjectParameters containing:
                - years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented: float
                - capability_cap_ai_randd_speedup: float (the fixed AI R&D speedup cap)
        """
        self.params = params if params is not None else USProjectParameters()
        self._capability_cap: Optional[Dict[str, Any]] = None
        self._trajectory: Optional[Dict[str, Any]] = None

    def compute_capability_cap(
        self,
        years: List[float],
        agreement_start_year: float,
        proxy_project: ProxyProject
    ) -> Dict[str, Any]:
        """
        Compute the capability cap (AI R&D speedup cap) over time.

        The capability cap is:
        - Before evaluation-based enforcement: proxy project's capabilities
        - After evaluation-based enforcement: max(fixed_cap, proxy_project_capabilities)

        Args:
            years: List of years for the time series
            agreement_start_year: The year when the AI slowdown agreement starts
            proxy_project: ProxyProject instance with computed trajectory

        Returns:
            Dictionary with:
                - 'years': List of years
                - 'capability_cap': List of AI R&D speedup cap values at each year
        """
        if not years:
            self._capability_cap = {'years': [], 'capability_cap': []}
            return self._capability_cap

        # When evaluation-based cap becomes enforceable
        eval_cap_start_year = (
            agreement_start_year +
            self.params.years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented
        )

        fixed_cap = self.params.capability_cap_ai_randd_speedup

        capability_caps = []

        for year in years:
            # Get proxy project's capabilities at this time
            proxy_speedup = proxy_project.get_speedup_at_time(year)

            # Default to 1.0 (no speedup) if proxy trajectory not available
            if proxy_speedup is None:
                proxy_speedup = 1.0

            if year < eval_cap_start_year:
                # Before evaluation-based enforcement:
                # Cap is whatever the proxy project can achieve
                cap = proxy_speedup
            else:
                # After evaluation-based enforcement:
                # Cap is the max of fixed cap and proxy project capabilities
                cap = max(fixed_cap, proxy_speedup)

            capability_caps.append(cap)

        self._capability_cap = {
            'years': list(years),
            'capability_cap': capability_caps
        }
        return self._capability_cap

    def compute_capability_cap_from_arrays(
        self,
        years: List[float],
        proxy_speedups: List[float],
        agreement_start_year: float
    ) -> Dict[str, Any]:
        """
        Compute the capability cap from arrays of years and proxy speedups.

        This is a convenience method that accepts arrays directly instead of
        requiring a ProxyProject instance. Useful when speedup data comes from
        Monte Carlo results or other sources.

        The capability cap is:
        - Before evaluation-based enforcement: proxy project's capabilities
        - After evaluation-based enforcement: max(fixed_cap, proxy_project_capabilities)

        Args:
            years: List of years for the time series
            proxy_speedups: List of proxy project AI R&D speedup values at each year
            agreement_start_year: The year when the AI slowdown agreement starts

        Returns:
            Dictionary with:
                - 'years': List of years
                - 'capability_cap': List of AI R&D speedup cap values at each year
        """
        if not years or not proxy_speedups:
            self._capability_cap = {'years': [], 'capability_cap': []}
            return self._capability_cap

        # When evaluation-based cap becomes enforceable
        eval_cap_start_year = (
            agreement_start_year +
            self.params.years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented
        )

        fixed_cap = self.params.capability_cap_ai_randd_speedup

        capability_caps = []

        for i, year in enumerate(years):
            proxy_speedup = proxy_speedups[i] if i < len(proxy_speedups) else 1.0

            if year < eval_cap_start_year:
                # Before evaluation-based enforcement:
                # Cap is whatever the proxy project can achieve
                cap = proxy_speedup
            else:
                # After evaluation-based enforcement:
                # Cap is the max of fixed cap and proxy project capabilities
                cap = max(fixed_cap, proxy_speedup)

            capability_caps.append(cap)

        self._capability_cap = {
            'years': list(years),
            'capability_cap': capability_caps
        }
        return self._capability_cap

    def compute_trajectory(
        self,
        years: List[float],
        compute: List[float],
        takeoff_model: Any,
        human_labor: Optional[List[float]] = None,
        agreement_start_year: Optional[float] = None,
        proxy_project: Optional[ProxyProject] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute the US project trajectory with the capability cap applied.

        This runs the takeoff model with the capability cap passed in, which
        upper-bounds progress at each time step. The result is the maximum
        capabilities the largest US AI company can achieve without exceeding
        the capability cap.

        Args:
            years: List of years for the time series
            compute: List of compute values (uncapped - the US company's actual compute)
            takeoff_model: TakeoffModel instance for computing trajectories
            human_labor: Optional list of human labor values. If None, uses default.
            agreement_start_year: The year when the AI slowdown agreement starts.
                Required if capability_cap hasn't been computed yet.
            proxy_project: ProxyProject instance with computed trajectory.
                Required if capability_cap hasn't been computed yet.

        Returns:
            Dictionary with trajectory data including 'speedup' or None if failed
        """
        # Compute capability cap if not already done
        if self._capability_cap is None:
            if agreement_start_year is None or proxy_project is None:
                raise ValueError(
                    "Must provide agreement_start_year and proxy_project if "
                    "capability_cap hasn't been computed yet"
                )
            self.compute_capability_cap(years, agreement_start_year, proxy_project)

        if human_labor is None:
            human_labor = [100.0] * len(years)

        # Get the capability cap values (AI R&D speedup limits)
        # These need to be converted to progress values for the takeoff model
        # The capability_cap is in terms of AI R&D speedup, which corresponds to
        # progress in the takeoff model
        cap_years = self._capability_cap['years']
        cap_values = self._capability_cap['capability_cap']

        # Interpolate capability cap to match the input years
        capability_cap_interpolated = np.interp(years, cap_years, cap_values)

        # Run trajectory with capability cap
        # Note: The TakeoffModel needs to be updated to accept capability_cap
        # For now, we use predict_trajectory_deterministic and pass capability_cap
        self._trajectory = takeoff_model.predict_trajectory_deterministic(
            time=years,
            human_labor=human_labor,
            compute=compute,
            capability_cap=list(capability_cap_interpolated)
        )

        if self._trajectory:
            self._trajectory['capability_cap'] = self._capability_cap

        return self._trajectory

    @property
    def capability_cap(self) -> Optional[Dict[str, Any]]:
        """Get the computed capability cap."""
        return self._capability_cap

    @property
    def trajectory(self) -> Optional[Dict[str, Any]]:
        """Get the computed trajectory."""
        return self._trajectory

    def get_capability_cap_at_time(self, year: float) -> Optional[float]:
        """
        Get the capability cap at a specific time.

        Args:
            year: The year to query

        Returns:
            The capability cap (AI R&D speedup limit) at that time,
            or None if not computed
        """
        if self._capability_cap is None:
            return None

        years = self._capability_cap.get('years', [])
        caps = self._capability_cap.get('capability_cap', [])

        if not years or not caps:
            return None

        # Interpolate to find cap at the requested year
        return float(np.interp(year, years, caps))

    def get_speedup_at_time(self, year: float) -> Optional[float]:
        """
        Get the AI R&D speedup at a specific time from the computed trajectory.

        Args:
            year: The year to query

        Returns:
            The AI R&D speedup at that time, or None if trajectory not computed
        """
        if self._trajectory is None or 'speedup' not in self._trajectory:
            return None

        years = self._trajectory.get('trajectory_times', [])
        speedups = self._trajectory.get('speedup', [])

        if not years or not speedups:
            return None

        # Interpolate to find speedup at the requested year
        return float(np.interp(year, years, speedups))
