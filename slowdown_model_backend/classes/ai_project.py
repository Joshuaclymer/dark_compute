"""
AI Project - Base class for AI projects with incremental simulation.

Projects are initialized and then advanced step-by-step via the update() method.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Add ai-futures-calculator to path for importing ProgressModelIncremental
_AI_FUTURES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ai-futures-calculator')
if _AI_FUTURES_PATH not in sys.path:
    sys.path.insert(0, _AI_FUTURES_PATH)
from progress_model_incremental import ProgressModelIncremental
import model_config as cfg

@dataclass
class MilestoneInfo:
    """Information about a milestone."""
    time: Optional[float] = None
    speedup: Optional[float] = None  # AI R&D speedup at milestone
    progress: Optional[float] = None  # Progress level at milestone
    ai_research_taste_sd: Optional[float] = None  # SD of AI research taste


@dataclass
class TakeoffCurve:
    """Data structure for takeoff curve results."""
    times: List[float]
    speedups: List[float]
    milestones: Dict[str, MilestoneInfo]  # {name: MilestoneInfo}
    # Additional time series for milestone tracking
    progress_history: List[float] = field(default_factory=list)
    ai_research_taste_sd_history: List[float] = field(default_factory=list)

    # Milestone full names
    MILESTONE_NAMES = {
        'AC': 'Automated Coder',
        'SAR': 'Superhuman AI Researcher',
        'SIAR': 'SuperIntelligent AI Researcher',
        'STRAT-AI': 'Strategic AI',
        'TED-AI': 'Top-Expert-Dominating AI',
        'ASI': 'Artificial SuperIntelligence',
    }

class AIProject:
    """
    Base class for AI projects with incremental simulation.

    Projects are initialized and then advanced step-by-step via the update() method.
    """


    def __init__(self, progress_model: ProgressModelIncremental):
        """
        Initialize the AI project.

        Args:
            progress_model: A ProgressModelIncremental instance (already calibrated)
        """
        self.progress_model = progress_model

        # Input histories for compute and labor (public attributes)
        self.time_history: List[float] = []
        self.human_labor_history: List[float] = []
        self.inference_compute_history: List[float] = []
        self.experiment_compute_history: List[float] = []
        self.training_compute_history: List[float] = []

    @abstractmethod
    def update_compute_and_labor(
        self,
        current_year: float,
        increment: float
    ) -> None:
        """
        Abstract method to update human_labor, inference_compute, experiment_compute, and training_compute.

        Args:
            current_year: The current year in the simulation
            increment: Time step in years
        """
        pass

    def update_takeoff_progress(self, year: float) -> None:
        """
        Advance the takeoff progress simulation to the specified year.

        Uses the most recent values from the history lists (which should be populated
        by subclasses before calling this method).

        Args:
            year: The year to advance the simulation to

        Raises:
            ValueError: If histories are empty (no values have been recorded)
        """
        if not self.human_labor_history:
            raise ValueError("Histories are empty. Call a method to populate histories before update_takeoff_progress.")

        # Get the most recent values from histories
        human_labor = self.human_labor_history[-1]
        inference_compute = self.inference_compute_history[-1]
        experiment_compute = self.experiment_compute_history[-1]
        training_compute = self.training_compute_history[-1]

        # Record time
        self.time_history.append(year)

        # Call progress_model.increment with the most recent values
        self.progress_model.increment(
            year,
            human_labor,
            inference_compute,
            experiment_compute,
            training_compute
        )

    def get_takeoff_curve(self) -> TakeoffCurve:
        """
        Get the takeoff curve data including times, AI R&D speedups, and milestones.

        Milestones tracked:
        - AC (Automated Coder): Progress reaches progress_at_aa threshold
        - SAR (Superhuman AI Researcher): AI research taste SD reaches TOP_RESEARCHER_SD (99.9th percentile)
        - SIAR (SuperIntelligent AI Researcher): AI research taste SD reaches 3x TOP_RESEARCHER_SD
        - STRAT-AI (Strategic AI): AI research taste SD reaches TOP_RESEARCHER_SD * (1 + strat_ai_m2b)
        - TED-AI (Top-Expert-Dominating AI): AI research taste SD reaches baseline * (1 + ted_ai_m2b)
        - ASI (Artificial SuperIntelligence): AI research taste SD reaches baseline * (1 + ted_ai_m2b + 2)
        - AIR-Nx: AI R&D speedup milestones (5x, 25x, 250x, 2000x, 10000x)

        Returns:
            TakeoffCurve with times, speedups, progress, AI research taste SD, and milestones
        """
        times = []
        speedups = []
        progress_history = []
        ai_research_taste_sd_history = []

        # Get milestone thresholds from model parameters
        progress_at_aa = self.progress_model.params.progress_at_aa
        strat_ai_m2b = getattr(self.progress_model.params, 'strat_ai_m2b', 2.0)
        ted_ai_m2b = getattr(self.progress_model.params, 'ted_ai_m2b', 3.0)

        # Track milestone states
        milestones: Dict[str, MilestoneInfo] = {}

        # AIR speedup milestone thresholds
        air_speedup_thresholds = {
            'AIR-5x': 5,
            'AIR-25x': 25,
            'AIR-250x': 250,
            'AIR-2000x': 2000,
            'AIR-10000x': 10000,
        }
        air_milestone_reached = {name: False for name in air_speedup_thresholds}

        # Capability milestones based on progress and AI research taste SD
        ac_reached = False
        sar_reached = False
        siar_reached = False
        strat_ai_reached = False
        ted_ai_reached = False
        asi_reached = False

        # SD thresholds for capability milestones
        sar_sd_threshold = cfg.TOP_RESEARCHER_SD  # 99.9th percentile (~3.09 SD)
        siar_sd_threshold = cfg.TOP_RESEARCHER_SD * 3.0
        strat_ai_sd_threshold = cfg.TOP_RESEARCHER_SD * (1.0 + strat_ai_m2b)

        # TED-AI and ASI baseline: will be set to SD at AC time if SAR would be reached before AC
        ted_ai_baseline_sd = cfg.TOP_RESEARCHER_SD  # Default, may be updated

        # Iterate through time_history (simulation times) rather than progress_model._history
        # to ensure we're using the correct time range for the covert project
        for i in range(len(self.time_history)):
            time = self.time_history[i]
            human_labor = self.human_labor_history[i]
            inference_compute = self.inference_compute_history[i]
            experiment_compute = self.experiment_compute_history[i]

            # Find the matching state in progress_model._history by time
            # The progress model history is indexed differently than our histories
            state = None
            for hist_state in self.progress_model._history:
                if abs(hist_state.time - time) < 0.01:  # Allow small tolerance
                    state = hist_state
                    break

            if state is None:
                # If no matching state, use the most recent state before this time
                for hist_state in reversed(self.progress_model._history):
                    if hist_state.time <= time:
                        state = hist_state
                        break

            if state is None:
                continue

            # Temporarily set state to compute metrics at that point
            original_progress = self.progress_model._state.progress
            original_rs = self.progress_model._state.research_stock
            self.progress_model._state.progress = state.progress
            self.progress_model._state.research_stock = state.research_stock

            metrics = self.progress_model.get_metrics(
                human_labor,
                inference_compute,
                experiment_compute,
                0.0
            )
            speedup = metrics.ai_sw_progress_mult_ref_present_day
            progress = state.progress
            ai_research_taste_sd = metrics.ai_research_taste_sd

            # Restore state
            self.progress_model._state.progress = original_progress
            self.progress_model._state.research_stock = original_rs

            times.append(time)
            speedups.append(speedup)
            progress_history.append(progress)
            ai_research_taste_sd_history.append(ai_research_taste_sd)

            # Check AC milestone (progress-based)
            if not ac_reached and progress_at_aa is not None and progress >= progress_at_aa:
                ac_reached = True
                crossing_time = self._interpolate_crossing_time(
                    times, progress_history, progress_at_aa, 'linear'
                )
                milestones['AC'] = MilestoneInfo(
                    time=crossing_time,
                    speedup=self._interp_at_time(crossing_time, times, speedups),
                    progress=progress_at_aa,
                    ai_research_taste_sd=self._interp_at_time(crossing_time, times, ai_research_taste_sd_history)
                )
                # Update TED-AI baseline if SAR experiment selection was reached before AC
                sd_at_ac = milestones['AC'].ai_research_taste_sd
                if sd_at_ac is not None and sd_at_ac >= sar_sd_threshold:
                    ted_ai_baseline_sd = sd_at_ac

            # Check SAR milestone (requires AC + SAR-level experiment selection skill)
            # SAR is achieved when BOTH AC is reached AND ai_research_taste_sd >= TOP_RESEARCHER_SD
            if not sar_reached and ac_reached and ai_research_taste_sd >= sar_sd_threshold:
                sar_reached = True
                # SAR time is the later of AC time and when SD threshold is crossed
                sd_crossing_time = self._interpolate_crossing_time(
                    times, ai_research_taste_sd_history, sar_sd_threshold, 'exponential'
                )
                sar_time = max(milestones['AC'].time, sd_crossing_time) if sd_crossing_time else milestones['AC'].time
                milestones['SAR'] = MilestoneInfo(
                    time=sar_time,
                    speedup=self._interp_at_time(sar_time, times, speedups),
                    progress=self._interp_at_time(sar_time, times, progress_history),
                    ai_research_taste_sd=self._interp_at_time(sar_time, times, ai_research_taste_sd_history)
                )

            # Check SIAR milestone (requires AC + SIAR-level experiment selection skill)
            if not siar_reached and ac_reached and ai_research_taste_sd >= siar_sd_threshold:
                siar_reached = True
                sd_crossing_time = self._interpolate_crossing_time(
                    times, ai_research_taste_sd_history, siar_sd_threshold, 'exponential'
                )
                siar_time = max(milestones['AC'].time, sd_crossing_time) if sd_crossing_time else milestones['AC'].time
                milestones['SIAR'] = MilestoneInfo(
                    time=siar_time,
                    speedup=self._interp_at_time(siar_time, times, speedups),
                    progress=self._interp_at_time(siar_time, times, progress_history),
                    ai_research_taste_sd=self._interp_at_time(siar_time, times, ai_research_taste_sd_history)
                )

            # Check STRAT-AI milestone (SD threshold, no AC requirement)
            if not strat_ai_reached and ai_research_taste_sd >= strat_ai_sd_threshold:
                strat_ai_reached = True
                crossing_time = self._interpolate_crossing_time(
                    times, ai_research_taste_sd_history, strat_ai_sd_threshold, 'exponential'
                )
                milestones['STRAT-AI'] = MilestoneInfo(
                    time=crossing_time,
                    speedup=self._interp_at_time(crossing_time, times, speedups),
                    progress=self._interp_at_time(crossing_time, times, progress_history),
                    ai_research_taste_sd=strat_ai_sd_threshold
                )

            # Check TED-AI milestone (SD threshold based on baseline)
            ted_ai_sd_threshold = ted_ai_baseline_sd * (1.0 + ted_ai_m2b)
            if not ted_ai_reached and ai_research_taste_sd >= ted_ai_sd_threshold:
                ted_ai_reached = True
                crossing_time = self._interpolate_crossing_time(
                    times, ai_research_taste_sd_history, ted_ai_sd_threshold, 'exponential'
                )
                milestones['TED-AI'] = MilestoneInfo(
                    time=crossing_time,
                    speedup=self._interp_at_time(crossing_time, times, speedups),
                    progress=self._interp_at_time(crossing_time, times, progress_history),
                    ai_research_taste_sd=ted_ai_sd_threshold
                )

            # Check ASI milestone (2 M2Bs above TED-AI)
            asi_sd_threshold = ted_ai_baseline_sd * (1.0 + ted_ai_m2b + 2.0)
            if not asi_reached and ai_research_taste_sd >= asi_sd_threshold:
                asi_reached = True
                crossing_time = self._interpolate_crossing_time(
                    times, ai_research_taste_sd_history, asi_sd_threshold, 'exponential'
                )
                milestones['ASI'] = MilestoneInfo(
                    time=crossing_time,
                    speedup=self._interp_at_time(crossing_time, times, speedups),
                    progress=self._interp_at_time(crossing_time, times, progress_history),
                    ai_research_taste_sd=asi_sd_threshold
                )

            # Check AIR speedup milestones
            for name, threshold in air_speedup_thresholds.items():
                if not air_milestone_reached[name] and speedup >= threshold:
                    air_milestone_reached[name] = True
                    crossing_time = self._interpolate_crossing_time(
                        times, speedups, threshold, 'exponential'
                    )
                    milestones[name] = MilestoneInfo(
                        time=crossing_time,
                        speedup=threshold,
                        progress=self._interp_at_time(crossing_time, times, progress_history),
                        ai_research_taste_sd=self._interp_at_time(crossing_time, times, ai_research_taste_sd_history)
                    )

        return TakeoffCurve(
            times=times,
            speedups=speedups,
            milestones=milestones,
            progress_history=progress_history,
            ai_research_taste_sd_history=ai_research_taste_sd_history
        )

    def _interpolate_crossing_time(
        self,
        times: List[float],
        values: List[float],
        threshold: float,
        interp_type: str = 'linear'
    ) -> Optional[float]:
        """
        Find the time when a value series crosses a threshold.

        Args:
            times: Time series
            values: Value series (same length as times)
            threshold: Threshold to find crossing for
            interp_type: 'linear' or 'exponential' interpolation

        Returns:
            Interpolated crossing time, or None if threshold not crossed
        """
        if len(times) < 2 or len(values) < 2:
            return times[-1] if times else None

        for i in range(1, len(values)):
            if values[i] >= threshold and values[i-1] < threshold:
                prev_time = times[i-1]
                curr_time = times[i]
                prev_val = values[i-1]
                curr_val = values[i]

                if interp_type == 'exponential' and prev_val > 0 and curr_val > 0:
                    # Linear interpolation in log space
                    log_prev = np.log(prev_val)
                    log_curr = np.log(curr_val)
                    log_thresh = np.log(threshold)
                    if log_curr != log_prev:
                        frac = (log_thresh - log_prev) / (log_curr - log_prev)
                    else:
                        frac = 0.5
                else:
                    # Linear interpolation
                    if curr_val != prev_val:
                        frac = (threshold - prev_val) / (curr_val - prev_val)
                    else:
                        frac = 0.5

                return prev_time + frac * (curr_time - prev_time)

        # If threshold is already exceeded at start or never crossed
        if values and values[-1] >= threshold:
            return times[-1]
        return None

    def _interp_at_time(
        self,
        target_time: Optional[float],
        times: List[float],
        values: List[float]
    ) -> Optional[float]:
        """Interpolate a value at a specific time."""
        if target_time is None or not times or not values:
            return None
        return float(np.interp(target_time, times, values))
    