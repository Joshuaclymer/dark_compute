#!/usr/bin/env python3
"""
Test script that plots the takeoff curve (AI R&D speedup over time with milestones).

This creates a plot similar to the slowdownPlot in the frontend, showing:
- AI R&D speedup on a log scale (y-axis)
- Time in years (x-axis)
- Milestone markers (AIR-5x, AIR-25x, etc.)
"""

import sys
import os

# Add paths for imports
_CLASSES_PATH = os.path.dirname(__file__)
_BACKEND_PATH = os.path.dirname(_CLASSES_PATH)
_ROOT_PATH = os.path.dirname(_BACKEND_PATH)
_AI_FUTURES_PATH = os.path.join(_ROOT_PATH, 'ai-futures-calculator')

if _AI_FUTURES_PATH not in sys.path:
    sys.path.insert(0, _AI_FUTURES_PATH)
if _CLASSES_PATH not in sys.path:
    sys.path.insert(0, _CLASSES_PATH)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress_model_incremental import ProgressModelIncremental
from progress_model import ProgressModel, Parameters, TimeSeriesData
import model_config as cfg
from ai_project import AIProject, TakeoffCurve, MilestoneInfo


def load_time_series() -> TimeSeriesData:
    """Load the default input_data.csv time series."""
    csv_path = os.path.join(_AI_FUTURES_PATH, 'input_data.csv')
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )


def create_progress_model(time_series: TimeSeriesData, start_year: float, end_year: float):
    """Create and calibrate a ProgressModelIncremental."""
    # Create params with linear interpolation
    params_dict = dict(cfg.DEFAULT_PARAMETERS)
    params_dict['automation_interp_type'] = 'linear'
    params = Parameters(**{
        k: v for k, v in params_dict.items()
        if k in Parameters.__dataclass_fields__
    })

    # Change to ai-futures-calculator directory for calibration
    original_cwd = os.getcwd()
    try:
        os.chdir(_AI_FUTURES_PATH)
        progress_model = ProgressModelIncremental(
            params=params,
            initial_time_series=time_series,
            initial_progress=0.0,
            time_range=[start_year, end_year],
            skip_calibration=False
        )
    finally:
        os.chdir(original_cwd)

    return progress_model


def compute_training_compute_trajectory(time_series: TimeSeriesData):
    """
    Compute absolute training compute values by integrating the growth rate.

    The growth rate is in OOMs/year, so we integrate:
    log10(TC(t)) = log10(TC(t0)) + integral(growth_rate, t0, t)

    Returns:
        Array of absolute training compute values aligned with time_series.time
    """
    # Base training compute at the start of time series (2012)
    # This value doesn't matter much since the model only uses growth rates,
    # but we set it to a reasonable historical value
    base_training_compute_ooms = 24.0  # 1e24 FLOP

    times = time_series.time
    growth_rates = time_series.training_compute_growth_rate

    # Integrate growth rate to get training compute OOMs at each time point
    training_compute_ooms = np.zeros(len(times))
    training_compute_ooms[0] = base_training_compute_ooms

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        # Average growth rate over the interval
        avg_growth_rate = 0.5 * (growth_rates[i-1] + growth_rates[i])
        training_compute_ooms[i] = training_compute_ooms[i-1] + avg_growth_rate * dt

    return 10 ** training_compute_ooms


def get_inputs_at_time(time_series: TimeSeriesData, t: float, training_compute_trajectory: np.ndarray):
    """Get interpolated input values at time t."""
    human_labor = np.exp(np.interp(t, time_series.time, np.log(time_series.L_HUMAN)))
    inference_compute = np.exp(np.interp(t, time_series.time, np.log(time_series.inference_compute)))
    experiment_compute = np.exp(np.interp(t, time_series.time, np.log(time_series.experiment_compute)))
    # Training compute: interpolate from pre-computed trajectory
    training_compute = np.exp(np.interp(t, time_series.time, np.log(training_compute_trajectory)))
    return human_labor, inference_compute, experiment_compute, training_compute


def run_simulation():
    """Run the simulation and return takeoff curve data."""
    print("Loading time series data...")
    time_series = load_time_series()

    start_year = 2026.0
    end_year = 2040.0
    increment = 0.1

    print(f"Creating progress model (start={start_year}, end={end_year})...")
    progress_model = create_progress_model(time_series, start_year, end_year)

    # Compute training compute trajectory by integrating growth rates
    training_compute_trajectory = compute_training_compute_trajectory(time_series)

    # Store initial inputs
    human_labor, inference_compute, experiment_compute, training_compute = \
        get_inputs_at_time(time_series, start_year, training_compute_trajectory)
    progress_model._input_history.append((
        start_year, human_labor, inference_compute, experiment_compute, training_compute
    ))

    # Create AIProject
    project = AIProject(progress_model=progress_model, increment=increment)

    print(f"Initial state: time={project.current_time}, progress={progress_model.progress:.2f}")

    # Run simulation
    num_steps = int((end_year - start_year) / increment)
    print(f"Running {num_steps} time steps...")

    for i in range(num_steps):
        t = start_year + (i + 1) * increment
        # Get inputs from time series (scaled with growth)
        human_labor, inference_compute, experiment_compute, training_compute = \
            get_inputs_at_time(time_series, min(t, time_series.time.max()), training_compute_trajectory)

        project.update_takeoff_progress(
            human_labor=human_labor,
            inference_compute=inference_compute,
            experiment_compute=experiment_compute,
            training_compute=training_compute
        )

    print(f"Final state: time={project.current_time:.1f}, progress={progress_model.progress:.2f}")

    # Get takeoff curve
    takeoff_curve = project.get_takeoff_curve()

    return takeoff_curve


def run_full_model_simulation():
    """Run the full ProgressModel and return takeoff curve data for comparison."""
    print("Loading time series data...")
    time_series = load_time_series()

    start_year = 2026.0
    end_year = 2040.0

    # Create params with linear interpolation
    params_dict = dict(cfg.DEFAULT_PARAMETERS)
    params_dict['automation_interp_type'] = 'linear'
    params = Parameters(**{
        k: v for k, v in params_dict.items()
        if k in Parameters.__dataclass_fields__
    })

    # Change to ai-futures-calculator directory for calibration
    original_cwd = os.getcwd()
    try:
        os.chdir(_AI_FUTURES_PATH)

        print("Creating and running full ProgressModel...")
        full_model = ProgressModel(params, time_series)
        time_range = [time_series.time.min(), time_series.time.max()]
        times, progress, rs = full_model.compute_progress_trajectory(time_range, initial_progress=0.0)

    finally:
        os.chdir(original_cwd)

    # Extract speedups from results
    results = full_model.results
    all_times = np.array(results['times'])
    all_speedups = np.array(results['ai_sw_progress_mult_ref_present_day'])

    # Filter to the range we care about
    mask = (all_times >= start_year) & (all_times <= end_year)
    filtered_times = all_times[mask].tolist()
    filtered_speedups = all_speedups[mask].tolist()

    # Extract milestones from full model results
    milestones = {}
    full_milestones = results.get('milestones', {})

    # Map to the milestone names we use
    milestone_mapping = {
        'AIR-5x': 'AIR-5x',
        'AIR-25x': 'AIR-25x',
        'AIR-250x': 'AIR-250x',
        'AIR-2000x': 'AIR-2000x',
        'AIR-10000x': 'AIR-10000x',
    }

    for our_name, full_name in milestone_mapping.items():
        if full_name in full_milestones and full_milestones[full_name].get('time') is not None:
            milestone_time = full_milestones[full_name]['time']
            milestone_speedup = full_milestones[full_name].get('progress_multiplier',
                                                               full_milestones[full_name].get('target', 0))
            if start_year <= milestone_time <= end_year:
                milestones[our_name] = MilestoneInfo(
                    time=milestone_time,
                    speedup=milestone_speedup
                )

    print(f"Full model: {len(filtered_times)} time points from {filtered_times[0]:.1f} to {filtered_times[-1]:.1f}")

    return TakeoffCurve(
        times=filtered_times,
        speedups=filtered_speedups,
        milestones=milestones
    )


def plot_takeoff_curve(takeoff_curve, full_model_curve=None):
    """Plot the takeoff curve similar to slowdownPlot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors
    incr_color = '#5B8DBE'  # Blue for incremental model
    full_color = '#C77CAA'  # Purple/pink for full model

    # Plot full model curve first (if provided) so it's behind
    if full_model_curve is not None:
        ax.plot(
            full_model_curve.times,
            full_model_curve.speedups,
            color=full_color,
            linewidth=4,
            linestyle='-',
            label='Full ProgressModel',
            alpha=0.6
        )

        # Plot full model milestones
        for name, data in full_model_curve.milestones.items():
            if data.time is not None and data.speedup is not None:
                ax.scatter(
                    [data.time],
                    [data.speedup],
                    s=80,
                    color=full_color,
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=4,
                    marker='s'  # Square markers for full model
                )

    # Plot the incremental model trajectory
    ax.plot(
        takeoff_curve.times,
        takeoff_curve.speedups,
        color=incr_color,
        linewidth=2,
        linestyle='--',
        label='Incremental Model (AIProject)',
        alpha=0.9
    )

    # Plot incremental model milestones
    for name, data in takeoff_curve.milestones.items():
        if data.time is not None and data.speedup is not None:
            ax.scatter(
                [data.time],
                [data.speedup],
                s=100,
                color=incr_color,
                edgecolors='white',
                linewidths=2,
                zorder=5
            )
            # Add label
            ax.annotate(
                name,
                (data.time, data.speedup),
                textcoords='offset points',
                xytext=(0, 12),
                ha='center',
                fontsize=10,
                color=incr_color,
                fontweight='bold'
            )

    # Configure axes
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('AI R&D Speedup', fontsize=12)

    title = 'AI Takeoff Trajectory'
    if full_model_curve is not None:
        title += ' (Incremental vs Full Model Comparison)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim(takeoff_curve.times[0], takeoff_curve.times[-1])

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Legend
    ax.legend(loc='upper left')

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("TAKEOFF CURVE TEST")
    print("=" * 60)

    # Run incremental model simulation
    print("\n--- Running Incremental Model (AIProject) ---")
    takeoff_curve = run_simulation()

    # Run full model simulation
    print("\n--- Running Full ProgressModel ---")
    full_model_curve = run_full_model_simulation()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Incremental Model (AIProject) ---")
    print(f"Trajectory: {len(takeoff_curve.times)} time points")
    print(f"  Start: {takeoff_curve.times[0]:.1f} (speedup: {takeoff_curve.speedups[0]:.2f}x)")
    print(f"  End: {takeoff_curve.times[-1]:.1f} (speedup: {takeoff_curve.speedups[-1]:.2f}x)")

    print(f"\nMilestones reached: {len(takeoff_curve.milestones)}")
    for name, data in sorted(takeoff_curve.milestones.items(), key=lambda x: x[1].time if x[1].time else float('inf')):
        time_str = f"{data.time:.2f}" if data.time else "N/A"
        speedup_str = f"{data.speedup:.2f}" if data.speedup else "N/A"
        print(f"  {name}: {time_str} ({speedup_str}x)")

    print("\n--- Full ProgressModel ---")
    print(f"Trajectory: {len(full_model_curve.times)} time points")
    print(f"  Start: {full_model_curve.times[0]:.1f} (speedup: {full_model_curve.speedups[0]:.2f}x)")
    print(f"  End: {full_model_curve.times[-1]:.1f} (speedup: {full_model_curve.speedups[-1]:.2f}x)")

    print(f"\nMilestones reached: {len(full_model_curve.milestones)}")
    for name, data in sorted(full_model_curve.milestones.items(), key=lambda x: x[1].time if x[1].time else float('inf')):
        time_str = f"{data.time:.2f}" if data.time else "N/A"
        speedup_str = f"{data.speedup:.2f}" if data.speedup else "N/A"
        print(f"  {name}: {time_str} ({speedup_str}x)")

    # Compare milestone times
    print("\n--- Milestone Comparison ---")
    all_milestones = set(takeoff_curve.milestones.keys()) | set(full_model_curve.milestones.keys())
    for name in sorted(all_milestones):
        incr_data = takeoff_curve.milestones.get(name)
        full_data = full_model_curve.milestones.get(name)
        incr_time = incr_data.time if incr_data else None
        full_time = full_data.time if full_data else None
        if incr_time and full_time:
            diff = incr_time - full_time
            print(f"  {name}: Incr={incr_time:.2f}, Full={full_time:.2f}, Diff={diff:+.2f} years")
        elif incr_time:
            print(f"  {name}: Incr={incr_time:.2f}, Full=N/A")
        elif full_time:
            print(f"  {name}: Incr=N/A, Full={full_time:.2f}")

    # Plot
    print("\nGenerating comparison plot...")
    fig = plot_takeoff_curve(takeoff_curve, full_model_curve)

    # Save plot
    output_path = os.path.join(_CLASSES_PATH, 'takeoff_curve_test.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Don't show to avoid hanging
    # plt.show()


if __name__ == "__main__":
    main()
