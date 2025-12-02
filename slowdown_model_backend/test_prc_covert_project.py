#!/usr/bin/env python3
"""
Test script for the SlowdownModel - PRC Covert AI Project Takeoff Curve

This script runs a simulation of the PRC Covert AI Project and plots the
takeoff curve (AI R&D speedup over time with milestones).
"""

import sys
import os

# Set up paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_AI_FUTURES_DIR = os.path.join(_ROOT_DIR, 'ai-futures-calculator')
_DARK_COMPUTE_DIR = os.path.join(_ROOT_DIR, 'black_project_backend')

# Add paths for imports - root dir first for absolute imports like black_project_backend.X
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _AI_FUTURES_DIR not in sys.path:
    sys.path.insert(0, _AI_FUTURES_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _DARK_COMPUTE_DIR not in sys.path:
    sys.path.insert(0, _DARK_COMPUTE_DIR)

# Change to ai-futures-calculator for calibration
original_cwd = os.getcwd()
os.chdir(_AI_FUTURES_DIR)

import numpy as np
import matplotlib.pyplot as plt
from progress_model import load_time_series_data, TimeSeriesData, Parameters

from slowdown_model import SlowdownModel, SlowdownSimulation
from slowdown_model_paramaters import (
    SlowdownModelParameters,
    SlowdownSimulationSettings,
    TakeoverRiskParameters,
    ProxyProjectParameters,
    SoftwareProliferationParameters,
    USProjectParameters,
)
from black_project_backend.black_project_parameters import (
    CovertProjectProperties,
    CovertProjectParameters,
)
from classes.ai_project import TakeoffCurve


def load_default_time_series() -> TimeSeriesData:
    """Load the default input_data.csv time series."""
    csv_path = os.path.join(_AI_FUTURES_DIR, 'input_data.csv')
    return load_time_series_data(csv_path)


def create_default_parameters() -> SlowdownModelParameters:
    """Create default SlowdownModelParameters for testing."""
    # All fields now have defaults, so we can just instantiate with no args
    # and then customize the settings we care about
    params = SlowdownModelParameters()

    # Customize simulation settings
    params.slowdown_simulation_settings.start_agreement_at_specific_year = 2030
    params.slowdown_simulation_settings.num_years_to_simulate = 10.0
    params.slowdown_simulation_settings.time_step_years = 0.1
    params.slowdown_simulation_settings.num_simulations = 1
    params.slowdown_simulation_settings.present_year = 2026
    params.slowdown_simulation_settings.end_year = 2040

    # Customize covert project properties
    params.covert_project_properties.run_a_covert_project = True
    params.covert_project_properties.proportion_of_initial_compute_stock_to_divert = 0.05
    params.covert_project_properties.datacenter_construction_labor = 10000
    params.covert_project_properties.years_before_agreement_year_prc_starts_building_covert_datacenters = 1
    params.covert_project_properties.build_a_covert_fab = True
    params.covert_project_properties.covert_fab_operating_labor = 550
    params.covert_project_properties.covert_fab_construction_labor = 250
    params.covert_project_properties.covert_fab_proportion_of_prc_lithography_scanners_devoted = 0.1

    # Customize covert project parameters
    params.covert_project_parameters.p_project_exists = 0.2

    return params


def plot_takeoff_curve(takeoff_curve: TakeoffCurve, title: str = "PRC Covert AI Project - Takeoff Curve"):
    """Plot the takeoff curve (AI R&D speedup over time)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Main trajectory
    ax.plot(
        takeoff_curve.times,
        takeoff_curve.speedups,
        color='#E74C3C',
        linewidth=2.5,
        label='PRC Covert Project AI R&D Speedup',
    )

    # Milestone markers with different styles
    milestone_colors = {
        'AC': '#27AE60',      # Green - Automated Coder
        'SAR': '#3498DB',     # Blue - Superhuman AI Researcher
        'SIAR': '#9B59B6',    # Purple - SuperIntelligent AI Researcher
        'STRAT-AI': '#F39C12', # Orange - Strategic AI
        'TED-AI': '#E91E63',  # Pink - Top-Expert-Dominating AI
        'ASI': '#1ABC9C',     # Teal - Artificial SuperIntelligence
    }

    air_color = '#95A5A6'  # Gray for AIR speedup milestones

    for name, info in takeoff_curve.milestones.items():
        if info.time is not None and info.speedup is not None:
            # Determine color
            if name in milestone_colors:
                color = milestone_colors[name]
                marker = 'o'
                size = 120
            else:
                color = air_color
                marker = 's'
                size = 80

            # Plot marker
            ax.scatter(
                [info.time],
                [info.speedup],
                s=size,
                color=color,
                edgecolors='white',
                linewidths=2,
                marker=marker,
                zorder=5,
                label=f'{name} ({info.time:.2f})',
            )

            # Add annotation
            offset_y = 15 if info.speedup > 10 else 10
            ax.annotate(
                name,
                (info.time, info.speedup),
                textcoords='offset points',
                xytext=(0, offset_y),
                ha='center',
                fontsize=9,
                fontweight='bold',
                color=color,
            )

    # Configure axes
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('AI R&D Speedup (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Set y-axis limits to show relevant range
    min_speedup = min(takeoff_curve.speedups) if takeoff_curve.speedups else 1
    max_speedup = max(takeoff_curve.speedups) if takeoff_curve.speedups else 100
    ax.set_ylim(min_speedup * 0.8, max_speedup * 2)

    # Legend (only show unique entries, sorted by time)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig


def plot_compute_trajectory(times, experiment_compute, title: str = "PRC Covert AI Project - Compute Trajectory"):
    """Plot the compute trajectory (experiment compute over time)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter out zero values for better plotting
    non_zero_indices = [i for i, ec in enumerate(experiment_compute) if ec > 0]

    if non_zero_indices:
        filtered_times = [times[i] for i in non_zero_indices]
        filtered_compute = [experiment_compute[i] for i in non_zero_indices]

        ax.plot(
            filtered_times,
            filtered_compute,
            color='#3498DB',
            linewidth=2.5,
            label='Experiment Compute (H100e)',
        )

        ax.scatter(
            filtered_times[0],
            filtered_compute[0],
            s=100,
            color='#27AE60',
            edgecolors='white',
            linewidths=2,
            marker='o',
            zorder=5,
            label=f'Start: {filtered_times[0]:.2f} ({filtered_compute[0]:.2e} H100e)',
        )

        max_idx = np.argmax(filtered_compute)
        ax.scatter(
            filtered_times[max_idx],
            filtered_compute[max_idx],
            s=100,
            color='#E74C3C',
            edgecolors='white',
            linewidths=2,
            marker='o',
            zorder=5,
            label=f'Peak: {filtered_times[max_idx]:.2f} ({filtered_compute[max_idx]:.2e} H100e)',
        )

    # Configure axes
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Experiment Compute (H100e, log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    return fig


def run_test():
    """Run the SlowdownModel simulation and plot the PRC covert project takeoff curve."""
    print("=" * 70)
    print("PRC COVERT AI PROJECT - TAKEOFF CURVE TEST")
    print("=" * 70)

    # Create parameters
    print("\n--- Creating Model Parameters ---")
    params = create_default_parameters()
    print(f"Agreement start year: {params.slowdown_simulation_settings.start_agreement_at_specific_year}")
    print(f"Present year: {params.slowdown_simulation_settings.present_year}")
    print(f"End year: {params.slowdown_simulation_settings.end_year}")
    print(f"Time step: {params.slowdown_simulation_settings.time_step_years} years")

    # Create and run the model
    print("\n--- Running SlowdownModel Simulation ---")
    model = SlowdownModel(params)

    # Run a single median simulation for deterministic results
    covert_projects, detectors = model.run_median_simulation()

    # Get the PRC covert project
    prc_project = covert_projects.get("prc_covert_project")
    if prc_project is None:
        print("ERROR: PRC covert project not found!")
        return None

    print(f"Project name: {prc_project.name}")
    print(f"Time history length: {len(prc_project.time_history)}")
    print(f"Compute history length: {len(prc_project.experiment_compute_history)}")

    # Debug: print time history range
    if prc_project.time_history:
        print(f"Time history range: {prc_project.time_history[0]:.2f} to {prc_project.time_history[-1]:.2f}")

    # Debug: Print compute history
    print("\n--- Compute History (first 10 non-zero entries) ---")
    non_zero_count = 0
    for i, (t, ec) in enumerate(zip(prc_project.time_history, prc_project.experiment_compute_history)):
        if ec > 0 and non_zero_count < 10:
            print(f"  Year {t:.2f}: experiment_compute = {ec:.2e} H100e")
            non_zero_count += 1
        if non_zero_count == 10:
            break

    # Debug: print progress model history range
    if prc_project.progress_model._history:
        print(f"\nProgress model history length: {len(prc_project.progress_model._history)}")
        print(f"Progress model history times: {prc_project.progress_model._history[0].time:.2f} to {prc_project.progress_model._history[-1].time:.2f}")

    # Get takeoff curve
    print("\n--- Getting Takeoff Curve ---")
    takeoff_curve = prc_project.get_takeoff_curve()

    # Print summary
    print(f"\nTrajectory: {len(takeoff_curve.times)} time points")
    if takeoff_curve.times:
        print(f"  Start: {takeoff_curve.times[0]:.2f} (speedup: {takeoff_curve.speedups[0]:.4f}x)")
        print(f"  End: {takeoff_curve.times[-1]:.2f} (speedup: {takeoff_curve.speedups[-1]:.4f}x)")

    print(f"\nMilestones reached: {len(takeoff_curve.milestones)}")
    for name, info in sorted(takeoff_curve.milestones.items(),
                             key=lambda x: x[1].time if x[1].time else float('inf')):
        time_str = f"{info.time:.2f}" if info.time else "N/A"
        speedup_str = f"{info.speedup:.2f}" if info.speedup else "N/A"
        progress_str = f"{info.progress:.2f}" if info.progress else "N/A"
        full_name = TakeoffCurve.MILESTONE_NAMES.get(name, name)
        print(f"  {name} ({full_name}): Year {time_str}, Speedup {speedup_str}x, Progress {progress_str}")

    # Print compute history summary
    print("\n--- Compute History Summary ---")
    if prc_project.experiment_compute_history:
        print(f"  Initial experiment compute: {prc_project.experiment_compute_history[0]:.2e} H100e")
        print(f"  Final experiment compute: {prc_project.experiment_compute_history[-1]:.2e} H100e")
        max_compute = max(prc_project.experiment_compute_history)
        print(f"  Max experiment compute: {max_compute:.2e} H100e")

    # Plot takeoff curve (note: currently has issues with progress model integration)
    print("\n--- Generating Takeoff Curve Plot ---")
    fig_takeoff = plot_takeoff_curve(takeoff_curve)
    output_path_takeoff = os.path.join(_SCRIPT_DIR, 'prc_covert_project_takeoff.png')
    fig_takeoff.savefig(output_path_takeoff, dpi=150, bbox_inches='tight')
    print(f"Saved takeoff curve to: {output_path_takeoff}")

    # Plot compute trajectory (this shows the covert project's compute buildup correctly)
    print("\n--- Generating Compute Trajectory Plot ---")
    fig_compute = plot_compute_trajectory(
        prc_project.time_history,
        prc_project.experiment_compute_history
    )
    output_path_compute = os.path.join(_SCRIPT_DIR, 'prc_covert_project_compute.png')
    fig_compute.savefig(output_path_compute, dpi=150, bbox_inches='tight')
    print(f"Saved compute trajectory to: {output_path_compute}")

    # Also display if in interactive mode
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except:
        pass

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return takeoff_curve


if __name__ == "__main__":
    os.chdir(original_cwd)  # Restore original directory before running
    os.chdir(_AI_FUTURES_DIR)  # Then change to ai-futures for calibration

    takeoff_curve = run_test()

    # Restore original directory
    os.chdir(original_cwd)
