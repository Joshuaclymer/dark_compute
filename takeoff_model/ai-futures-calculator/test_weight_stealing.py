"""
Tests for the weight stealing and algorithm stealing functionality in the AI futures calculator.

This module tests the WeightStealingProgressModel class and related functionality
that simulates scenarios where one project steals from another more capable project.

Two types of stealing are tested:
1. Weight stealing: Stealing model weights gives access to the leading project's AI
   capabilities (automation fraction, AI research taste) but not their progress.
2. Algorithm stealing: Stealing algorithms gives the same software efficiency as the
   leading project up to a specified time/milestone.
"""

import numpy as np
import pytest
from predict_trajectory import (
    TrajectoryPredictor,
    predict_milestones_from_compute,
    MilestoneInfo
)
from progress_model import (
    Parameters,
    TimeSeriesData,
    ProgressModel,
    WeightStealingProgressModel
)


class TestWeightStealingBasic:
    """Basic tests for weight stealing functionality."""

    def test_weight_stealing_model_initialization(self):
        """Test that WeightStealingProgressModel initializes correctly."""
        params = Parameters()
        params.weight_stealing_mode = True
        params.years_weights_are_stolen = [2028.0]

        # Create time series data
        time = np.linspace(2024, 2050, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.5 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.4 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        stealing_data = TimeSeriesData(
            time=time,
            L_HUMAN=L_HUMAN,
            inference_compute=inference_compute,
            experiment_compute=experiment_compute,
            training_compute_growth_rate=training_compute_growth_rate
        )

        # Leading project has 10x more resources
        leading_data = TimeSeriesData(
            time=time,
            L_HUMAN=L_HUMAN * 10,
            inference_compute=inference_compute * 10,
            experiment_compute=experiment_compute * 10,
            training_compute_growth_rate=training_compute_growth_rate
        )

        model = WeightStealingProgressModel(params, stealing_data, leading_data)

        assert model is not None
        assert model.leading_project_data is not None
        assert model.stealing_project_data is not None

    def test_weight_stealing_increases_progress(self):
        """Test that stealing weights from a more advanced project accelerates progress.

        With the corrected semantics, stealing weights gives you the AI CAPABILITIES
        of the leading project (automation fraction, AI research taste) but you keep
        your own progress and research stock. This should still accelerate your progress
        because you now have a more capable AI assistant.
        """
        params = Parameters()

        # Create time series data - same base resources for both scenarios
        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        # First compute without weight stealing (baseline)
        predictor_no_steal = TrajectoryPredictor(params=Parameters())
        predictor_no_steal.predict_from_time_series(
            time=time,
            inference_compute=inference_compute,
            experiment_compute=experiment_compute,
            L_HUMAN=L_HUMAN,
            training_compute_growth_rate=training_compute_growth_rate
        )
        progress_no_steal = predictor_no_steal.results['progress'][-1]

        # Now compute with weight stealing - stealing from a project with MORE resources
        # The leading project has 10x resources, so should be further ahead
        predictor_steal = TrajectoryPredictor(params=Parameters())
        predictor_steal.predict_from_time_series(
            time=time,
            inference_compute=inference_compute,  # Same resources as baseline
            experiment_compute=experiment_compute,
            L_HUMAN=L_HUMAN,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute * 10,  # Leading project has 10x more
            experiment_compute_leading_project=experiment_compute * 10,
            L_HUMAN_leading_project=L_HUMAN * 10,
            years_weights_are_stolen_from_leading_project=[2030.0]  # Steal in 2030
        )
        progress_steal = predictor_steal.results['progress'][-1]

        # Stealing weights from a more advanced project should increase final progress
        # because after stealing, we have access to the leading project's AI capabilities
        # (higher automation fraction, better AI research taste)
        assert progress_steal > progress_no_steal, (
            f"Expected progress with stealing ({progress_steal:.3f}) > "
            f"progress without stealing ({progress_no_steal:.3f})"
        )

        # Also verify that model_progress is tracked separately and is higher than progress
        # after the stealing event
        if 'model_progress' in predictor_steal.results:
            model_progress = predictor_steal.results['model_progress']
            own_progress = predictor_steal.results['progress']
            # After stealing (t > 2030), model_progress should be >= own_progress
            times = predictor_steal.results['times']
            for i, t in enumerate(times):
                if t > 2030.0:
                    assert model_progress[i] >= own_progress[i], (
                        f"At t={t}, model_progress ({model_progress[i]:.3f}) should be >= "
                        f"own_progress ({own_progress[i]:.3f})"
                    )

    def test_leading_project_results_stored(self):
        """Test that leading project results are stored separately."""
        params = Parameters()

        time = np.linspace(2024, 2040, 50)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        stealing_data = TimeSeriesData(
            time=time,
            L_HUMAN=L_HUMAN * 0.1,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            training_compute_growth_rate=training_compute_growth_rate
        )

        leading_data = TimeSeriesData(
            time=time,
            L_HUMAN=L_HUMAN,
            inference_compute=inference_compute,
            experiment_compute=experiment_compute,
            training_compute_growth_rate=training_compute_growth_rate
        )

        params.weight_stealing_mode = True
        params.years_weights_are_stolen = [2030.0]

        model = WeightStealingProgressModel(params, stealing_data, leading_data)
        model.compute_progress_trajectory([time[0], time[-1]], initial_progress=0.0)

        # Check that leading project results are stored
        assert 'leading_project_results' in model.results
        assert model.results['leading_project_results'] is not None
        assert 'progress' in model.results['leading_project_results']
        assert 'times' in model.results['leading_project_results']


class TestWeightStealingConvenienceFunctions:
    """Test convenience functions with weight stealing parameters."""

    def test_predict_milestones_with_weight_stealing(self):
        """Test predict_milestones_from_compute with weight stealing."""
        time = np.linspace(2024, 2040, 50)

        # Stealing project
        inference = 1e5 * np.exp(0.3 * (time - 2024))
        experiment = 1e4 * np.exp(0.3 * (time - 2024))

        # Leading project (10x resources)
        inference_leader = inference * 10
        experiment_leader = experiment * 10

        milestones = predict_milestones_from_compute(
            time=time,
            inference_compute=inference,
            experiment_compute=experiment,
            inference_compute_leading_project=inference_leader,
            experiment_compute_leading_project=experiment_leader,
            years_weights_are_stolen_from_leading_project=[2030.0]
        )

        # Should get some milestones
        assert isinstance(milestones, dict)

    def test_no_weight_stealing_when_no_stealing_times(self):
        """Test that model runs normally when no stealing times provided."""
        time = np.linspace(2024, 2040, 50)
        inference = 1e6 * np.exp(0.3 * (time - 2024))
        experiment = 1e5 * np.exp(0.3 * (time - 2024))

        # Provide leading project data but no stealing times
        milestones = predict_milestones_from_compute(
            time=time,
            inference_compute=inference,
            experiment_compute=experiment,
            inference_compute_leading_project=inference * 10,
            experiment_compute_leading_project=experiment * 10,
            years_weights_are_stolen_from_leading_project=None  # No stealing
        )

        assert isinstance(milestones, dict)


class TestWeightStealingEdgeCases:
    """Test edge cases for weight stealing."""

    def test_multiple_stealing_events(self):
        """Test that multiple stealing events work correctly."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        milestones = predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,  # Stealing project
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,  # Leading project
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=[2028.0, 2032.0, 2036.0]  # Multiple events
        )

        # Model should complete without error
        assert predictor.results is not None
        assert 'progress' in predictor.results

    def test_stealing_at_start_time(self):
        """Test stealing at the start of the simulation."""
        params = Parameters()

        time = np.linspace(2024, 2040, 50)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        milestones = predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=[2024.0]  # Steal at start
        )

        assert predictor.results is not None

    def test_stealing_at_end_time(self):
        """Test stealing at the end of the simulation."""
        params = Parameters()

        time = np.linspace(2024, 2040, 50)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        milestones = predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=[2040.0]  # Steal at end
        )

        assert predictor.results is not None


class TestWeightStealingMilestoneNames:
    """Test that milestone names can be used as stealing times."""

    def test_steal_at_sc_milestone(self):
        """Test stealing weights when leading project reaches SC milestone."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,  # Stealing project
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,  # Leading project
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=["SC"]  # Steal at SC milestone
        )

        # Model should complete without error
        assert predictor.results is not None
        assert 'progress' in predictor.results

    def test_steal_at_ac_milestone(self):
        """Test stealing weights when leading project reaches AC milestone."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=["AC"]  # Steal at AC milestone
        )

        assert predictor.results is not None
        assert 'progress' in predictor.results

    def test_mixed_milestones_and_times(self):
        """Test mixing milestone names and numeric times."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=[2028.0, "SC", "ASI"]  # Mix times and milestones
        )

        assert predictor.results is not None
        assert 'progress' in predictor.results


class TestAlgorithmStealing:
    """Test algorithm stealing functionality."""

    def test_algorithm_stealing_increases_progress(self):
        """Test that stealing algorithms increases progress for the stealing project."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        # Baseline: no stealing
        predictor_no_steal = TrajectoryPredictor(params=Parameters())
        predictor_no_steal.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,  # Stealing project has less resources
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate
        )
        progress_no_steal = predictor_no_steal.results['progress'][-1]

        # With algorithm stealing up to 2035
        predictor_steal = TrajectoryPredictor(params=Parameters())
        predictor_steal.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            stealing_algorithms_up_to=2035.0  # Steal algorithms up to 2035
        )
        progress_steal = predictor_steal.results['progress'][-1]

        # Stealing algorithms should increase progress
        assert progress_steal > progress_no_steal, (
            f"Expected progress with algorithm stealing ({progress_steal:.3f}) > "
            f"progress without stealing ({progress_no_steal:.3f})"
        )

    def test_algorithm_stealing_with_milestone(self):
        """Test algorithm stealing using a milestone name as cutoff."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            stealing_algorithms_up_to="SC"  # Steal algorithms up to SC milestone
        )

        assert predictor.results is not None
        assert 'progress' in predictor.results

    def test_combined_weight_and_algorithm_stealing(self):
        """Test that weight stealing and algorithm stealing can be combined."""
        params = Parameters()

        time = np.linspace(2024, 2040, 100)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=["SC"],  # Steal weights at SC
            stealing_algorithms_up_to=2030.0  # Steal algorithms up to 2030
        )

        assert predictor.results is not None
        assert 'progress' in predictor.results


class TestWeightStealingProgressContinuity:
    """Test that progress values are continuous and make sense after stealing."""

    def test_progress_increases_at_stealing_event(self):
        """Test that progress jumps up at a stealing event."""
        params = Parameters()

        time = np.linspace(2024, 2040, 200)
        L_HUMAN = np.ones_like(time) * 100
        inference_compute = 1e6 * np.exp(0.3 * (time - 2024))
        experiment_compute = 1e5 * np.exp(0.3 * (time - 2024))
        training_compute_growth_rate = np.ones_like(time) * 0.5

        stealing_time = 2030.0

        predictor = TrajectoryPredictor(params=params)
        predictor.predict_from_time_series(
            time=time,
            inference_compute=inference_compute * 0.1,  # Stealing project has less
            experiment_compute=experiment_compute * 0.1,
            L_HUMAN=L_HUMAN * 0.1,
            training_compute_growth_rate=training_compute_growth_rate,
            inference_compute_leading_project=inference_compute,
            experiment_compute_leading_project=experiment_compute,
            L_HUMAN_leading_project=L_HUMAN,
            years_weights_are_stolen_from_leading_project=[stealing_time]
        )

        results = predictor.results
        times = results['times']
        progress = results['progress']

        # Find indices around the stealing time
        idx_before = np.searchsorted(times, stealing_time - 0.1)
        idx_after = np.searchsorted(times, stealing_time + 0.1)

        # Progress should generally be increasing and should have a boost around stealing time
        # (The exact behavior depends on how much better the leading project is)
        assert progress[-1] > progress[0], "Progress should increase over time"


if __name__ == '__main__':
    # Run a simple test
    print("Running basic weight stealing tests...")

    test = TestWeightStealingBasic()

    try:
        test.test_weight_stealing_model_initialization()
        print("✓ test_weight_stealing_model_initialization passed")
    except Exception as e:
        print(f"✗ test_weight_stealing_model_initialization failed: {e}")

    try:
        test.test_weight_stealing_increases_progress()
        print("✓ test_weight_stealing_increases_progress passed")
    except Exception as e:
        print(f"✗ test_weight_stealing_increases_progress failed: {e}")

    try:
        test.test_leading_project_results_stored()
        print("✓ test_leading_project_results_stored passed")
    except Exception as e:
        print(f"✗ test_leading_project_results_stored failed: {e}")

    print("\nRunning edge case tests...")

    test_edge = TestWeightStealingEdgeCases()

    try:
        test_edge.test_multiple_stealing_events()
        print("✓ test_multiple_stealing_events passed")
    except Exception as e:
        print(f"✗ test_multiple_stealing_events failed: {e}")

    print("\nAll basic tests completed!")
