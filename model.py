from dataclasses import dataclass
from typing import Optional, List
from fab_model import CovertFab, PRCCovertFab, ProcessNode
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================================
# CLASSES
# ============================================================================

@dataclass
class CovertProjectStrategy:
    run_a_covert_project : bool

    # Covert fab
    build_a_covert_fab : bool
    covert_fab_operating_labor : Optional[int]
    covert_fab_construction_labor : Optional[int]
    covert_fab_process_node : Optional[ProcessNode]
    covert_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float]

@dataclass
class CovertProject:
    name : str
    h100e_over_time : dict[List[float], List[float]]  # year -> cumulative H100e produced
    covert_project_strategy : CovertProjectStrategy
    agreement_year : float
    construction_start_year : Optional[float] = None
    covert_fab : Optional[CovertFab] = None

    def __post_init__(self):
        # Initialize covert fab if strategy requires it and construction_start_year is set
        if (self.covert_project_strategy.build_a_covert_fab and
            self.construction_start_year is not None):
            self.covert_fab = PRCCovertFab(
                construction_start_year = self.construction_start_year,
                construction_labor = self.covert_project_strategy.covert_fab_construction_labor,
                process_node = self.covert_project_strategy.covert_fab_process_node,
                proportion_of_prc_lithography_scanners_devoted_to_fab = self.covert_project_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted,
                operation_labor = self.covert_project_strategy.covert_fab_operating_labor,
                agreement_year = self.agreement_year
            )


@dataclass
class DetectorStrategy:
    placeholder : float # just leave this incomplete for now

@dataclass
class BeliefsAboutProject:
    p_project_exists : float
    p_covert_fab_exists : float
    project_strategy_conditional_on_existence : CovertProjectStrategy
    distribution_over_compute_operation : List[dict[List[float], List[float]]]  # List[year -> cumulative H100e produced]

@dataclass
class Detector:
    name : str
    strategy : DetectorStrategy
    beliefs_about_projects : dict[str, List[BeliefsAboutProject]]  # project name -> BeliefsAboutProject at each time step

# ============================================================================
# DEFAULTS

default_us_detection_strategy = DetectorStrategy(
    placeholder = 0.0
)
default_prc_covert_project_strategy = CovertProjectStrategy(
    run_a_covert_project = True,
    build_a_covert_fab = True,
    covert_fab_operating_labor = 728,
    covert_fab_construction_labor = 448,
    covert_fab_process_node = "best_available_indigenously",
    covert_fab_proportion_of_prc_lithography_scanners_devoted = 0.102,
)

best_prc_covert_project_strategy = default_prc_covert_project_strategy

# ============================================================================
# SIMULATION: A simulation of a single rollout of a possible world
# ============================================================================

@dataclass
class Simulation:

    def __init__(
            self,
            year_us_prc_agreement_goes_into_force : float,
            covert_projects: dict[str, CovertProject],
            detectors: dict[str, Detector]
        ):
        self.year_us_prc_agreement_goes_into_force = year_us_prc_agreement_goes_into_force
        self.covert_projects = covert_projects
        self.detectors = detectors

    def run_simulation(self, end_year : float, increment: float = 0.01):

        # Store initial priors to prevent them from being overwritten
        # The LRs are cumulative, so we always multiply by the initial prior
        initial_priors = {}
        for detector_name, detector in self.detectors.items():
            initial_priors[detector_name] = {}
            for project_name, beliefs_by_year in detector.beliefs_about_projects.items():
                if self.year_us_prc_agreement_goes_into_force in beliefs_by_year:
                    initial_priors[detector_name][project_name] = {
                        'p_covert_fab_exists': beliefs_by_year[self.year_us_prc_agreement_goes_into_force].p_covert_fab_exists,
                        'p_project_exists': beliefs_by_year[self.year_us_prc_agreement_goes_into_force].p_project_exists
                    }

        # Run the simulation from the agreement year to the end year
        current_year = self.year_us_prc_agreement_goes_into_force
        while current_year <= end_year:
            for project in self.covert_projects.values():
                current_h100e = project.h100e_over_time.get(current_year - increment, 0.0)

                # Make updates pertaining to covert fabs
                if project.covert_fab is not None:
                    # h100e_produced_per_month returns monthly production, multiply by increment (in years) and months per year
                    updated_h100e = project.covert_fab.h100e_produced_per_month(current_year) * 12 * increment
                    project.h100e_over_time[current_year] = current_h100e + updated_h100e
                    # Update detector beliefs
                    likelihood_ratio = project.covert_fab.detection_likelihood_ratio(year=current_year)
                    for detector_name, detector in self.detectors.items():
                        # Use stored initial priors (LRs are cumulative, so always multiply by initial prior)
                        prior_p_covert_fab_exists = initial_priors[detector_name][project.name]['p_covert_fab_exists']
                        prior_p_covert_project_exists = initial_priors[detector_name][project.name]['p_project_exists']
                        updated_odds_of_covert_fab = prior_p_covert_fab_exists / (1 - prior_p_covert_fab_exists) * likelihood_ratio
                        updated_p_covert_fab_exists = updated_odds_of_covert_fab / (1 + updated_odds_of_covert_fab)
                        updated_odds_of_covert_project = prior_p_covert_project_exists / (1 - prior_p_covert_project_exists) * likelihood_ratio
                        updated_p_covert_project_exists = updated_odds_of_covert_project / (1 + updated_odds_of_covert_project)
                        detector.beliefs_about_projects[project.name][current_year] = BeliefsAboutProject(
                            p_project_exists=updated_p_covert_project_exists,
                            p_covert_fab_exists=updated_p_covert_fab_exists,
                            project_strategy_conditional_on_existence=detector.beliefs_about_projects[project.name][self.year_us_prc_agreement_goes_into_force].project_strategy_conditional_on_existence,
                            distribution_over_compute_operation=[]
                        )

            current_year += increment
        return self.covert_projects, self.detectors

# ============================================================================
# MODEL: Aggregates results across simulations
# ============================================================================

class Model:
    def __init__(
            self,
            year_us_prc_agreement_goes_into_force : float,
            end_year : float,
            increment: float,
        ):
        self.year_us_prc_agreement_goes_into_force = year_us_prc_agreement_goes_into_force
        self.end_year = end_year
        self.increment = increment

        self.initial_detectors = {
            "us_intelligence" : Detector(
                name = "us_intelligence",
                strategy = default_us_detection_strategy,
                beliefs_about_projects = {
                    "prc_covert_project" : {
                        year_us_prc_agreement_goes_into_force:
                            BeliefsAboutProject(
                            p_project_exists = 0.2,
                            p_covert_fab_exists = 0.1,
                            project_strategy_conditional_on_existence = best_prc_covert_project_strategy,
                            distribution_over_compute_operation = []
                    )
                }}
            )
        }

        self.simulation_results = []

    def _create_fresh_covert_projects(self):
        """Create a new set of covert projects with fresh random sampling.

        This method should be called for each simulation to ensure that
        random parameters (like detection times, localization years, etc.)
        are independently sampled for each simulation run.

        Returns:
            dict: Dictionary of covert project names to CovertProject instances
        """
        return {
            "prc_covert_project" : CovertProject(
                name = "prc_covert_project",
                h100e_over_time = {self.year_us_prc_agreement_goes_into_force: 0.0},
                covert_project_strategy = default_prc_covert_project_strategy,
                agreement_year = self.year_us_prc_agreement_goes_into_force,
                construction_start_year = self.year_us_prc_agreement_goes_into_force
            )
        }

    def run_simulations(self, num_simulations : int):

        for _ in range(num_simulations):
            simulation = Simulation(
                year_us_prc_agreement_goes_into_force = self.year_us_prc_agreement_goes_into_force,
                covert_projects = self._create_fresh_covert_projects(),  # Create fresh projects with new random sampling
                detectors = deepcopy(self.initial_detectors)
            )
            covert_projects, detectors = simulation.run_simulation(
                end_year = self.end_year,
                increment = self.increment
            )
            self.simulation_results.append((covert_projects, detectors))
        
    def plot_probability_and_compute_over_time(self, save_path=None):
        """
        Create a dual y-axis plot showing US probability and H100e count over time.

        Args:
            save_path: Optional path to save the plot. If None, only displays the plot.
        """
        if not self.simulation_results:
            print("No simulation results to display. Run simulations first.")
            return

        # Extract data from all simulations (limit to first 100 for visualization)
        # Structure: years -> list of (p_covert_fab_exists, h100e_count) for each simulation
        all_years = []
        us_probs_by_sim = []  # List of lists: [sim_idx][time_idx] -> probability
        h100e_by_sim = []     # List of lists: [sim_idx][time_idx] -> h100e count

        # Limit to first 100 simulations for cleaner visualization
        simulations_to_plot = self.simulation_results[:100]

        for covert_projects, detectors in simulations_to_plot:
            # Get US intelligence beliefs about PRC covert project
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]

            # Get H100e production from covert project
            h100e_over_time = covert_projects["prc_covert_project"].h100e_over_time

            # Extract time series data
            years = sorted(us_beliefs.keys())
            if not all_years:
                all_years = years

            us_probs = [us_beliefs[year].p_covert_fab_exists for year in years]
            h100e_counts = [h100e_over_time.get(year, 0.0) for year in years]

            us_probs_by_sim.append(us_probs)
            h100e_by_sim.append(h100e_counts)

        # Convert to numpy arrays for easier manipulation
        years_array = np.array(all_years)
        us_probs_array = np.array(us_probs_by_sim)  # Shape: (num_sims, num_timesteps)
        h100e_array = np.array(h100e_by_sim)        # Shape: (num_sims, num_timesteps)

        # Calculate medians and percentiles
        us_probs_median = np.median(us_probs_array, axis=0)
        us_probs_p25 = np.percentile(us_probs_array, 25, axis=0)
        us_probs_p75 = np.percentile(us_probs_array, 75, axis=0)
        h100e_median = np.median(h100e_array, axis=0)
        h100e_p25 = np.percentile(h100e_array, 25, axis=0)
        h100e_p75 = np.percentile(h100e_array, 75, axis=0)

        # Convert H100e to thousands for better readability
        h100e_array_thousands = h100e_array / 1e3
        h100e_median_thousands = h100e_median / 1e3
        h100e_p25_thousands = h100e_p25 / 1e3
        h100e_p75_thousands = h100e_p75 / 1e3

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot US probability on left y-axis
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('US Probability Covert Fab Exists', fontsize=12, fontweight='bold', color='darkblue')  # Match median line color

        # Plot individual simulations with thin, semi-transparent lines
        # Lower alpha creates darker layering effect where lines overlap
        for i in range(len(us_probs_by_sim)):
            ax1.plot(years_array, us_probs_array[i], color='lightblue', alpha=0.1, linewidth=0.8, zorder=1)

        # Plot filled region for 25th-75th percentiles
        ax1.fill_between(years_array, us_probs_p25, us_probs_p75, color='blue', alpha=0.2, zorder=2, label='25th-75th Percentile')

        # Plot median with bold line on top
        ax1.plot(years_array, us_probs_median, color='darkblue', linewidth=3.0, label='Median US probability of PRC project', zorder=100)

        ax1.tick_params(axis='y', labelcolor='darkblue')  # Match median line color
        ax1.set_ylim([0, 1])
        ax1.set_xlim([self.year_us_prc_agreement_goes_into_force, self.end_year])  # Span from start to end date
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Create second y-axis for H100e count
        ax2 = ax1.twinx()
        ax2.set_ylabel('H100e in PRC Covert Project', fontsize=12, fontweight='bold', color='darkred')  # Match median line color

        # Plot individual simulations with thin, semi-transparent lines
        # Lower alpha creates darker layering effect where lines overlap
        for i in range(len(h100e_by_sim)):
            ax2.plot(years_array, h100e_array_thousands[i], color='lightcoral', alpha=0.1, linewidth=0.8, zorder=1)

        # Plot filled region for 25th-75th percentiles
        ax2.fill_between(years_array, h100e_p25_thousands, h100e_p75_thousands, color='red', alpha=0.3, zorder=2, label='25th-75th Percentile', edgecolor='darkred', linewidth=0.5)

        # Plot median with bold line on top
        ax2.plot(years_array, h100e_median_thousands, color='darkred', linewidth=3.0, label='Median H100e count in PRC project', zorder=100)

        ax2.tick_params(axis='y', labelcolor='darkred')  # Match median line color
        ax2.set_ylim([0, 1000])  # Set maximum to 1000K

        # Format y-axis ticks as "100K", "200K", etc.
        ax2.set_yticks([0, 200, 400, 600, 800, 1000])
        ax2.set_yticklabels(['0', '200K', '400K', '600K', '800K', '1000K'])

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        legend.set_zorder(1000)

        # Add info text
        info_text = f'Showing {len(simulations_to_plot)} simulations (of {len(self.simulation_results)} total)'
        plt.figtext(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=9, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()

    def display_aggregate_statistics(self):
        """
        Display aggregate statistics across all simulations.

        Creates plots showing:
        - US probability and H100e count over time (dual y-axis)
        - Compute produced before detection analysis (three panels)
        """
        self.plot_probability_and_compute_over_time(save_path='simulation_runs.png')
        self.plot_compute_before_detection(save_path='compute_before_detection.png')

    def plot_compute_before_detection(self, detection_threshold=0.5, save_path='compute_before_detection.png'):
        """
        Create a three-panel plot showing:
        1. Distribution of compute produced before detection
        2. CCDF of compute produced before detection
        3. CCDF of operational time before detection

        Detection is defined as when US probability exceeds detection_threshold (default 0.5).

        Args:
            detection_threshold: Probability threshold for "detection" (default 0.5)
            save_path: Path to save the plot (default 'compute_before_detection.png')
        """
        if not self.simulation_results:
            print("No simulation results to display. Run simulations first.")
            return

        # Extract data from all simulations
        compute_at_detection = []  # H100e produced when detected
        operational_time_at_detection = []  # Years operational when detected

        for covert_projects, detectors in self.simulation_results:
            # Get US intelligence beliefs and H100e production
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            h100e_over_time = covert_projects["prc_covert_project"].h100e_over_time
            covert_fab = covert_projects["prc_covert_project"].covert_fab

            if covert_fab is None:
                continue

            # Find first year when US probability exceeds threshold
            years = sorted(us_beliefs.keys())
            detection_year = None
            for year in years:
                if us_beliefs[year].p_covert_fab_exists >= detection_threshold:
                    detection_year = year
                    break

            # Include all simulations with covert fabs (detected or not)
            if detection_year is not None:
                # Get compute produced at detection
                h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                compute_at_detection.append(h100e_at_detection)

                # Calculate operational time at detection
                # If detected during construction, operational time is 0
                # If detected after becoming operational, operational time is detection_year - operational_start
                construction_start = covert_fab.construction_start_year
                construction_duration = covert_fab.construction_duration
                operational_start = construction_start + construction_duration

                if detection_year >= operational_start:
                    operational_time = detection_year - operational_start
                else:
                    operational_time = 0.0  # Detected during construction

                operational_time_at_detection.append(operational_time)
            else:
                # If never detected, use final H100e value at end of simulation
                final_year = max(h100e_over_time.keys()) if h100e_over_time else None
                if final_year:
                    h100e_at_detection = h100e_over_time[final_year]
                    compute_at_detection.append(h100e_at_detection)

                    # Calculate operational time at end of simulation
                    construction_start = covert_fab.construction_start_year
                    construction_duration = covert_fab.construction_duration
                    operational_start = construction_start + construction_duration

                    if final_year >= operational_start:
                        operational_time = final_year - operational_start
                    else:
                        operational_time = 0.0  # Still in construction

                    operational_time_at_detection.append(operational_time)
                else:
                    # No H100e produced at all
                    compute_at_detection.append(0.0)
                    operational_time_at_detection.append(0.0)

        # Check if we have any data
        if not compute_at_detection:
            print("No covert fabs created in simulations. Try running more simulations or adjusting parameters.")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Panel 1: CCDF of compute produced before detection
        compute_sorted = np.sort(compute_at_detection)
        ccdf_compute = 1.0 - np.arange(1, len(compute_sorted) + 1) / len(compute_sorted)

        ax1.plot(compute_sorted, ccdf_compute, color='darkred', linewidth=2)

        # Only use log scale if there's sufficient range in the data
        if len(compute_sorted) > 0 and compute_sorted[-1] > 0:
            range_ratio = compute_sorted[-1] / max(compute_sorted[0], 1e-10)
            if range_ratio > 10:  # Use log scale only if range spans more than 1 order of magnitude
                ax1.set_xscale('log')

        ax1.set_xlabel('Total H100 equivalents produced (at detection or end)', fontsize=11)
        ax1.set_ylabel('P(Compute Produced ≥ x)', fontsize=11)
        ax1.set_title('CCDF of covert compute produced', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])

        # Set x-axis to start at 0 (or close to 0 for log scale)
        if len(compute_sorted) > 0:
            if ax1.get_xscale() == 'log':
                ax1.set_xlim([max(1, compute_sorted[0] * 0.5), None])
            else:
                ax1.set_xlim([0, None])

        # Add markers at 100K and 1M H100e
        target_computes = [100000, 1000000]  # 100K and 1M H100e
        for target_compute in target_computes:
            if len(compute_sorted) > 0 and compute_sorted[-1] > target_compute:
                # Find the CCDF value at target H100e
                idx = np.searchsorted(compute_sorted, target_compute)
                ccdf_at_target = 1.0 - idx / len(compute_sorted)

                # Add a marker point with label
                ax1.plot(target_compute, ccdf_at_target, 'o', color='black', markersize=8, zorder=10)
                label = f'100K' if target_compute == 100000 else f'1M'
                ax1.text(target_compute, ccdf_at_target, f'  ({label}, {ccdf_at_target*100:.1f}%)',
                        fontsize=10, verticalalignment='center', horizontalalignment='left')

        # Panel 2: CCDF of operational time before detection
        if operational_time_at_detection:
            op_time_sorted = np.sort(operational_time_at_detection)
            ccdf_op_time = 1.0 - np.arange(1, len(op_time_sorted) + 1) / len(op_time_sorted)

            ax2.plot(op_time_sorted, ccdf_op_time, color='darkred', linewidth=2)
            ax2.set_xlabel('Operational time before detection (years)', fontsize=11)
            ax2.set_ylabel('P(Operational time ≥ x)', fontsize=11)
            ax2.set_title('CCDF of operational time before detection', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0, 1])
        else:
            ax2.text(0.5, 0.5, 'No operational fabs detected',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_xlabel('Operational time before detection (years)', fontsize=11)
            ax2.set_ylabel('P(Operational time ≥ x)', fontsize=11)
            ax2.set_title('CCDF of operational time before detection', fontsize=10)

        try:
            plt.tight_layout()
        except ValueError:
            pass  # Skip tight_layout if it fails

        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()

    def plot_fab_characteristics(self, save_path: str = 'fab_characteristics.png'):
        """Plot process node distribution and percentage achieving >500K H100e (at detection or simulation end)."""
        if not self.simulation_results:
            print("No simulation results available. Run simulations first.")
            return

        detection_threshold = 0.5  # 50% probability threshold for detection
        target_compute = 500000  # 500K H100e threshold

        # Collect process nodes and whether each achieved >500K H100e before detection
        process_nodes = []
        achieved_500k_by_node = {}  # Dictionary mapping process node to list of booleans

        for covert_projects, detectors in self.simulation_results:
            prc_covert_project = covert_projects['prc_covert_project']
            if prc_covert_project and prc_covert_project.covert_fab:
                fab = prc_covert_project.covert_fab
                node = fab.process_node

                # Get US intelligence beliefs and H100e production
                us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
                h100e_over_time = prc_covert_project.h100e_over_time

                # Find first year when US probability exceeds threshold
                years = sorted(us_beliefs.keys())
                detection_year = None
                for year in years:
                    if us_beliefs[year].p_covert_fab_exists >= detection_threshold:
                        detection_year = year
                        break

                # Determine if this simulation achieved >500K H100e (at detection or end)
                achieved_500k = False
                if detection_year is not None:
                    h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                    achieved_500k = (h100e_at_detection >= target_compute)
                else:
                    # If never detected, check final H100e value
                    final_year = max(h100e_over_time.keys()) if h100e_over_time else None
                    if final_year:
                        h100e_at_end = h100e_over_time[final_year]
                        achieved_500k = (h100e_at_end >= target_compute)

                process_nodes.append(node)
                if node not in achieved_500k_by_node:
                    achieved_500k_by_node[node] = []
                achieved_500k_by_node[node].append(achieved_500k)

        if not process_nodes:
            print("No fab data available in simulations.")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Panel 1: Process node distribution (bar chart showing frequencies)
        from collections import Counter
        node_counts = Counter(process_nodes)
        total_count = len(process_nodes)

        # Define specific ordering: 130nm, 28nm, 14nm, 7nm
        # Map process node values to their nm size for sorting
        def get_nm_value(node):
            node_str = str(node)
            if 'nm' in node_str:
                return int(node_str.split('nm')[1].split(')')[0])
            return 999  # Unknown nodes go to the end

        # Get unique nodes and sort in descending order (130, 28, 14, 7)
        unique_nodes = list(set(process_nodes))
        nodes = sorted(unique_nodes, key=get_nm_value, reverse=True)
        frequencies = [node_counts[node] / total_count for node in nodes]

        # Convert node names to display format (e.g., "ProcessNode.nm28" -> "28nm")
        node_labels = []
        for node in nodes:
            node_str = str(node)
            if 'nm' in node_str:
                nm_value = node_str.split('nm')[1].split(')')[0]
                node_labels.append(f"{nm_value}nm")
            else:
                node_labels.append(node_str)

        ax1.bar(node_labels, frequencies, color='steelblue', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Process Node', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Distribution of Fab Process Nodes', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Panel 2: Percentage achieving >500K H100e by process node
        percent_500k_by_node = []
        for node in nodes:
            if node in achieved_500k_by_node and len(achieved_500k_by_node[node]) > 0:
                # Calculate percentage that achieved >500K H100e
                num_achieved = sum(achieved_500k_by_node[node])
                total = len(achieved_500k_by_node[node])
                percent = (num_achieved / total) * 100
                percent_500k_by_node.append(percent)
            else:
                percent_500k_by_node.append(0)

        ax2.bar(node_labels, percent_500k_by_node, color='steelblue', edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Process Node', fontsize=11)
        ax2.set_ylabel('% achieving >500K H100e', fontsize=11)
        ax2.set_title('% of times fab produces >500K H100e', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_ylim([0, 100])

        try:
            plt.tight_layout()
        except ValueError:
            pass  # Skip tight_layout if it fails

        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()

    def plot_process_node_success_proportion(self, save_path: str = 'process_node_success.png'):
        """Plot proportion of successful simulations (>500K H100e) by process node.

        Among all simulations that achieved >500K H100e (at detection or simulation end),
        what percentage came from each process node? These percentages sum to 100%.
        """
        if not self.simulation_results:
            print("No simulation results available. Run simulations first.")
            return

        detection_threshold = 0.5  # 50% probability threshold for detection
        target_compute = 500000  # 500K H100e threshold

        # Count successful simulations by process node
        success_count_by_node = {}  # Maps process node to count of successful simulations

        for covert_projects, detectors in self.simulation_results:
            prc_covert_project = covert_projects['prc_covert_project']
            if prc_covert_project and prc_covert_project.covert_fab:
                fab = prc_covert_project.covert_fab
                node = fab.process_node

                # Get US intelligence beliefs and H100e production
                us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
                h100e_over_time = prc_covert_project.h100e_over_time

                # Find first year when US probability exceeds threshold
                years = sorted(us_beliefs.keys())
                detection_year = None
                for year in years:
                    if us_beliefs[year].p_covert_fab_exists >= detection_threshold:
                        detection_year = year
                        break

                # Check if this simulation achieved >500K H100e (at detection or end)
                achieved_target = False
                if detection_year is not None:
                    h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                    achieved_target = (h100e_at_detection >= target_compute)
                else:
                    # If never detected, check final H100e value
                    final_year = max(h100e_over_time.keys()) if h100e_over_time else None
                    if final_year:
                        h100e_at_end = h100e_over_time[final_year]
                        achieved_target = (h100e_at_end >= target_compute)

                if achieved_target:
                    # This simulation succeeded
                    if node not in success_count_by_node:
                        success_count_by_node[node] = 0
                    success_count_by_node[node] += 1

        if not success_count_by_node:
            print("No simulations achieved >500K H100e.")
            return

        # Calculate total successful simulations
        total_successful = sum(success_count_by_node.values())

        # Define specific ordering: 130nm, 28nm, 14nm, 7nm
        def get_nm_value(node):
            node_str = str(node)
            if 'nm' in node_str:
                return int(node_str.split('nm')[1].split(')')[0])
            return 999

        # Get all nodes that had any successes and sort them
        nodes = sorted(success_count_by_node.keys(), key=get_nm_value, reverse=True)

        # Calculate proportions (as percentages of successful simulations)
        proportions = []
        for node in nodes:
            proportion = (success_count_by_node[node] / total_successful) * 100
            proportions.append(proportion)

        # Convert node names to display format
        node_labels = []
        for node in nodes:
            node_str = str(node)
            if 'nm' in node_str:
                nm_value = node_str.split('nm')[1].split(')')[0]
                node_labels.append(f"{nm_value}nm")
            else:
                node_labels.append(node_str)

        # Create single bar plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.bar(node_labels, proportions, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Process Node', fontsize=11)
        ax.set_ylabel('% of successful simulations', fontsize=11)
        ax.set_title('% of simulations where process node produces >500K H100e', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim([0, 100])  # Set y-axis limit to 100% since they sum to 100%

        try:
            plt.tight_layout()
        except ValueError:
            pass  # Skip tight_layout if it fails

        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Run simulations with default parameters
    model = Model(
        year_us_prc_agreement_goes_into_force=2030,
        end_year=2037,
        increment=0.1
    )

    print("Running simulations...")
    model.run_simulations(num_simulations=1000)

    print("Generating plots...")
    model.plot_probability_and_compute_over_time(save_path='simulation_runs.png')
    model.plot_compute_before_detection(save_path='compute_before_detection.png')
    model.plot_fab_characteristics(save_path='fab_characteristics.png')
    model.plot_process_node_success_proportion(save_path='process_node_success.png')

    print("Done! Plots saved to simulation_runs.png, compute_before_detection.png, fab_characteristics.png, and process_node_success.png")
