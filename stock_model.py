import numpy as np
import random
from typing import List, Dict
from abc import ABC, abstractmethod

# Constants for H100 chip (duplicated from fab_model to avoid circular import)
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per H100 chip (134 TFLOP/s FP16 * 16 bits)
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass (default, can be overridden)

class InitialPRCComputeStockParameters():

    # H100 power consumption
    h100_power_watts = 700  # Total power consumption of NVIDIA H100 GPU

    # PRC compute stock
    total_prc_compute_stock_in_2025 = 1e6
    energy_efficiency_relative_to_h100 = 0.5
    annual_growth_rate_of_prc_compute_stock = 2.4
    relative_sigma_of_prc_compute_stock = 0.1

    us_intelligence_median_error_in_estimate_of_prc_compute_stock = 0.07

    # Global compute stock parameters
    total_global_compute_in_2025 = 1e7
    annual_growth_rate_of_global_compute = 2.4
    relative_sigma_of_global_compute = 0.1

    median_unreported_compute_owned_by_non_prc_actors = 1e6
    relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

class SurvivalRateParameters():

    # Survival rate parameters
    initial_hazard_rate_p50 = 0.01
    increase_of_hazard_rate_per_year_p50 = 0.0035
    hazard_rate_p25_relative_to_p50 = 0.1
    hazard_rate_p75_relative_to_p50 = 6

def sample_initial_prc_compute_stock(year):
    """Sample initial PRC compute stock based on year and proportion diverted to covert project"""
    from util import sample_from_log_normal
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    median_total_stock = params.total_prc_compute_stock_in_2025 * (params.annual_growth_rate_of_prc_compute_stock ** years_since_2025)
    relative_sigma = params.relative_sigma_of_prc_compute_stock

    total_stock_sample = sample_from_log_normal(median_total_stock, relative_sigma)

    return total_stock_sample

def sample_us_estimate_of_prc_compute_stock(prc_compute_stock):
    # Calculate pdf of absolute relative error
    k = -np.log(0.5) / InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock

    u = np.random.uniform(0, 1)

    # Invert the CDF: P(|error|/actual <= x) = 1 - e^(-kx)
    # u = 1 - e^(-kx)
    # e^(-kx) = 1 - u
    # -kx = ln(1 - u)
    # x = -ln(1 - u) / k

    relative_error = -np.log(1 - u) / k

    # Randomly choose direction of error (overestimate or underestimate)
    error_sign = 1 if random.random() > 0.5 else -1
    relative_error = error_sign * relative_error

    # Apply error to actual count
    us_estimate = prc_compute_stock * (1 + relative_error)

    # Ensure estimate is non-negative
    return max(0, us_estimate)


def lr_from_prc_compute_accounting(reported_prc_compute_stock, optimal_diversion_proportion, us_estimate_of_prc_compute_stock):
    """Calculate likelihood ratio from global compute accounting"""
    # Consider two cases:
    # Case 1: Covert project exists of optimal size
    # Case 2: Covert project does not exist

    # Case 1: Covert project exists of optimal size
    # What is the probability that the us_estimate_of_prc_scanners would be what it is in this case?

    true_compute_stock_if_covert_project_exists = reported_prc_compute_stock / (1 - optimal_diversion_proportion)

    # Handle edge cases where the compute stock is zero or near zero
    if true_compute_stock_if_covert_project_exists < 1e-10 or reported_prc_compute_stock < 1e-10:
        return 1.0  # Neutral evidence if no scanners

    us_estimate_absolute_error_if_project_exists = abs(us_estimate_of_prc_compute_stock - true_compute_stock_if_covert_project_exists) / true_compute_stock_if_covert_project_exists

    # PDF of absolute error
    k = -np.log(0.5) / InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock
    p_observe_us_estimate_error_if_project_exists = k * np.exp(-k * us_estimate_absolute_error_if_project_exists)

    # Case 2: Fab does not exist
    true_prc_compute_stock_if_no_project_exists = reported_prc_compute_stock
    us_estimate_absolute_error_if_no_project_exists = abs(us_estimate_of_prc_compute_stock - true_prc_compute_stock_if_no_project_exists) / true_prc_compute_stock_if_no_project_exists

    p_observe_us_estimate_error_if_no_project_exists = k * np.exp(-k * us_estimate_absolute_error_if_no_project_exists)

    if p_observe_us_estimate_error_if_no_project_exists > 0:
        lr = p_observe_us_estimate_error_if_project_exists / p_observe_us_estimate_error_if_no_project_exists
    else:
        lr = 1e6  # Very large if no-fab scenario makes estimate very unlikely

    return lr

def sample_global_compute(year):
    """Sample global compute stock based on year"""
    from util import sample_from_log_normal
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    median_total_stock = params.total_global_compute_in_2025 * (params.annual_growth_rate_of_global_compute ** years_since_2025)
    relative_sigma = params.relative_sigma_of_global_compute

    global_compute = sample_from_log_normal(median_total_stock, relative_sigma)

    return global_compute

def sample_reported_global_compute(prc_compute_stock_diverted, global_compute):
    def _sample_unreported_compute_owned_by_non_prc_actors():
        """Sample unreported compute owned by non-PRC actors based on year"""
        from util import sample_from_log_normal
        params = InitialPRCComputeStockParameters
        median = params.median_unreported_compute_owned_by_non_prc_actors = 1e6
        relative_sigma = params.relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

        unreported_compute_owned_by_non_prc_actors = sample_from_log_normal(median, relative_sigma)

        return unreported_compute_owned_by_non_prc_actors
    
    unreported_compute_owned_by_non_prc_actors = _sample_unreported_compute_owned_by_non_prc_actors()

    reported_global_compute = global_compute - unreported_compute_owned_by_non_prc_actors - prc_compute_stock_diverted

    print(f"DEBUG: global_compute unreported by non PRC actors: {unreported_compute_owned_by_non_prc_actors}", flush=True)
    print(f"DEBUG: global_compute unreported by PRC: {prc_compute_stock_diverted}", flush=True)
    return reported_global_compute

def get_reported_global_compute_production(year, prc_compute_stock_diverted):
    total_global_compute = InitialPRCComputeStockParameters.total_global_compute_in_2025 * (InitialPRCComputeStockParameters.annual_growth_rate_of_global_compute ** (year - 2025))
    return total_global_compute - prc_compute_stock_diverted

def lr_from_global_compute_production_accounting(reported_historical_global_compute_production, reported_global_compute, reported_prc_compute_stock, optimal_diversion_proportion):

    discrepency = reported_global_compute - reported_historical_global_compute_production

    # Case 1: There's no covert PRC project
    params = InitialPRCComputeStockParameters
    relative_sigma = params.relative_sigma_unreported_compute_owned_by_non_prc_actors
    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(params.median_unreported_compute_owned_by_non_prc_actors)

    # Lognormal PDF
    if discrepency > 0:
        p_discrepency_if_no_prc_project = (1 / (discrepency * sigma_log * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(discrepency) - mu_log) / sigma_log)**2)
    else:
        p_discrepency_if_no_prc_project = 1e-10  # Very small probability for non-positive values
    

    # Case 2: There is a covert PRC project
    estimated_prc_compute_stock = reported_prc_compute_stock / (1 - optimal_diversion_proportion)
    estimated_computes_diverted_to_prc_project = estimated_prc_compute_stock - reported_prc_compute_stock
    estimated_compute_unreported_by_non_prc_actors = discrepency - estimated_computes_diverted_to_prc_project

    # Lognormal PDF
    if estimated_compute_unreported_by_non_prc_actors > 0:
        p_discrepency_if_prc_project = (1 / (estimated_compute_unreported_by_non_prc_actors * sigma_log * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(estimated_compute_unreported_by_non_prc_actors) - mu_log) / sigma_log)**2)
    else:
        p_discrepency_if_prc_project = 1e-10  # Very small probability for non-positive values

    lr = p_discrepency_if_prc_project / p_discrepency_if_no_prc_project

    return lr

def sample_hazard_rate_multiplier() -> float:
    """Sample a multiplier for hazard rates from a metalog distribution.

    Returns a multiplier that will be applied to both initial_hazard_rate and
    increase_of_hazard_rate_per_year to create correlated uncertainty.
    """
    from util import sample_from_metalog_3term_semi_bounded
    params = SurvivalRateParameters

    # Compute absolute percentiles from the ratios
    p25 = params.hazard_rate_p25_relative_to_p50
    p50 = 1.0  # Median multiplier is 1.0
    p75 = params.hazard_rate_p75_relative_to_p50

    # Sample multiplier from semi-bounded metalog distribution (lower bound 0, no upper bound)
    return sample_from_metalog_3term_semi_bounded(p25, p50, p75)

def sample_hazard_rates() -> tuple[float, float]:
    """Sample both initial hazard rate and increase rate using a common multiplier.

    Returns:
        tuple: (initial_hazard_rate, increase_of_hazard_rate_per_year)
    """
    params = SurvivalRateParameters
    multiplier = sample_hazard_rate_multiplier()

    initial_hazard_rate = params.initial_hazard_rate_p50 * multiplier
    increase_of_hazard_rate_per_year = params.increase_of_hazard_rate_per_year_p50 * multiplier

    return initial_hazard_rate, increase_of_hazard_rate_per_year


class Chip():
    def __init__(self, h100e_tpp_per_chip: float, W_of_energy_consumed: float, intra_chip_memory_bandwidth_tbps: float = 8, inter_chip_memory_bandwidth_tbps: float = 1.8):
        self.h100e_tpp_per_chip = h100e_tpp_per_chip
        self.W_of_energy_consumed = W_of_energy_consumed
        self.intra_chip_memory_bandwidth_tbps = intra_chip_memory_bandwidth_tbps
        self.inter_chip_memory_bandwidth_tbps = inter_chip_memory_bandwidth_tbps
    
class Compute():
    def __init__(self, chip_counts: Dict[Chip, float]):
        self.chip_counts = chip_counts  # Dict of Chip to number of chips

    def total_h100e_tpp(self) -> float:
        return sum(chip.h100e_tpp_per_chip * count for chip, count in self.chip_counts.items())

    def total_energy_requirements_GW(self) -> float:
        return sum(chip.W_of_energy_consumed * count for chip, count in self.chip_counts.items()) / 1e9  # Convert W to GW

    def energy_by_chip_efficiency(self) -> list[tuple[float, float]]:
        """Get energy consumption grouped by chip efficiency (H100e TPP per GW).

        Returns:
            List of (efficiency, energy_gw) tuples, sorted by efficiency (most efficient first)
            where efficiency = H100e TPP per GW
        """
        chip_data = []
        for chip, count in self.chip_counts.items():
            if chip.W_of_energy_consumed > 0:
                # Calculate efficiency: H100e TPP per GW
                energy_gw = (chip.W_of_energy_consumed * count) / 1e9
                h100e_tpp = chip.h100e_tpp_per_chip * count
                efficiency = h100e_tpp / energy_gw if energy_gw > 0 else 0
                chip_data.append((efficiency, energy_gw))

        # Sort by efficiency (highest first) so most efficient chips are at the bottom of the stack
        chip_data.sort(key=lambda x: x[0], reverse=True)
        return chip_data

    def energy_by_source(self, initial_stock_chips: set, fab_chips: set) -> tuple[float, float]:
        """Get energy consumption separated by source (initial stock vs fab-produced).

        Args:
            initial_stock_chips: Set of Chip objects that are from initial stock
            fab_chips: Set of Chip objects that are from fab production

        Returns:
            Tuple of (initial_stock_energy_gw, fab_energy_gw)
        """
        initial_energy = 0.0
        fab_energy = 0.0

        for chip, count in self.chip_counts.items():
            energy_gw = (chip.W_of_energy_consumed * count) / 1e9

            if chip in initial_stock_chips:
                initial_energy += energy_gw
            elif chip in fab_chips:
                fab_energy += energy_gw

        return (initial_energy, fab_energy)

class PRCDarkComputeStock():

    def __init__(self, agreement_year, proportion_of_initial_compute_stock_to_divert, optimal_proportion_of_initial_compute_stock_to_divert):
        self.agreement_year = agreement_year
        self.initial_prc_stock = sample_initial_prc_compute_stock(agreement_year)
        self.initial_prc_dark_compute = self.initial_prc_stock * proportion_of_initial_compute_stock_to_divert

        # dark_compute_added_per_year now stores chip dictionaries instead of single numbers
        # Structure: {year: {chip_id: {'count': float, 'h100_equivalence': float,
        #                                'energy_efficiency_relative_to_h100': float, 'bandwidth': float}}}
        self.dark_compute_added_per_year = {}

        # Initialize with a default chip representing the initial PRC dark compute stock
        params = InitialPRCComputeStockParameters
        self.dark_compute_added_per_year[agreement_year] = {
            'initial_prc_stock': {
                'count': 1.0,  # Normalized count
                'h100_equivalence': self.initial_prc_dark_compute,
                'energy_efficiency_relative_to_h100': params.energy_efficiency_relative_to_h100,
                'bandwidth': 1.8  # Default inter-chip bandwidth in tbps (H100-like)
            }
        } 
        # TEMPORARILY DISABLED FOR PERFORMANCE
        self.us_estimate_of_prc_stock = self.initial_prc_stock  # Skip sampling
        self.lr_from_prc_compute_accounting = 1.0  # No detection evidence
        self.global_compute = 0  # Skip global compute sampling
        self.reported_global_compute = 0  # Skip reported global compute
        self.reported_historical_global_compute_production = 0  # Skip historical production
        self.lr_from_global_compute_production_accounting = 1.0  # No detection evidence

        # # Original code (disabled):
        # self.us_estimate_of_prc_stock = sample_us_estimate_of_prc_compute_stock(self.initial_prc_stock)
        # self.lr_from_prc_compute_accounting = lr_from_prc_compute_accounting(
        #     reported_prc_compute_stock=self.initial_prc_stock - self.initial_prc_dark_compute,
        #     optimal_diversion_proportion=optimal_proportion_of_initial_compute_stock_to_divert,
        #     us_estimate_of_prc_compute_stock=self.us_estimate_of_prc_stock
        # )
        # self.global_compute = sample_global_compute(agreement_year)
        # print(f"DEBUG: global compute: {self.global_compute}", flush=True)

        # self.reported_global_compute = sample_reported_global_compute(self.initial_prc_dark_compute, self.global_compute)
        # print(f"DEBUG: reported global compute: {self.reported_global_compute}", flush=True)

        # self.reported_historical_global_compute_production = get_reported_global_compute_production(agreement_year, self.initial_prc_dark_compute)
        # self.lr_from_global_compute_production_accounting = lr_from_global_compute_production_accounting(
        #     reported_historical_global_compute_production= self.reported_historical_global_compute_production,
        #     reported_global_compute=self.reported_global_compute,
        #     reported_prc_compute_stock=self.initial_prc_stock - self.initial_prc_dark_compute,
        #     optimal_diversion_proportion=optimal_proportion_of_initial_compute_stock_to_divert
        # )
        # Sample both hazard rates using the correlated multiplier
        self.initial_hazard_rate, self.increase_in_hazard_rate_per_year = sample_hazard_rates()
    
    def add_dark_compute(self, year : float, additional_dark_compute : float):
        """Add dark compute for a year. This will create a default chip entry.

        Args:
            year: The year to add compute for
            additional_dark_compute: The H100-equivalent compute to add
        """
        params = InitialPRCComputeStockParameters
        if year not in self.dark_compute_added_per_year:
            self.dark_compute_added_per_year[year] = {}

        # Add as a new chip with default characteristics
        chip_id = f'added_year_{year}'
        self.dark_compute_added_per_year[year][chip_id] = {
            'count': 1.0,
            'h100_equivalence': additional_dark_compute,
            'energy_efficiency_relative_to_h100': params.energy_efficiency_relative_to_h100,
            'bandwidth': 1.8  # Default inter-chip bandwidth in tbps
        }
    
    def dark_compute_dead_and_alive(self, year : float):
        """Calculate total compute across all years up to the given year.

        Args:
            year: The year to calculate up to

        Returns:
            Compute object containing all chips (alive and dead)
        """
        # Aggregate chips by their properties to create Chip objects
        chip_counts = {}

        for y in self.dark_compute_added_per_year:
            if y <= year:
                for chip_id, chip_data in self.dark_compute_added_per_year[y].items():
                    # Calculate per-chip values
                    h100e_tpp_per_chip = chip_data['h100_equivalence'] / chip_data['count'] if chip_data['count'] > 0 else 0

                    # Calculate energy per chip
                    # h100e_tpp_per_chip is already in H100-equivalent TPP
                    # Convert to raw TPP, then to watts
                    tpp_per_chip = h100e_tpp_per_chip * H100_TPP_PER_CHIP
                    # Use H100 watts per TPP, adjusted by energy efficiency
                    watts_per_chip = tpp_per_chip * H100_WATTS_PER_TPP / chip_data['energy_efficiency_relative_to_h100']

                    # Create or find matching Chip object
                    chip = Chip(
                        h100e_tpp_per_chip=h100e_tpp_per_chip,
                        W_of_energy_consumed=watts_per_chip,
                        inter_chip_memory_bandwidth_tbps=chip_data['bandwidth']
                    )

                    # Add count to existing chip or create new entry
                    chip_counts[chip] = chip_counts.get(chip, 0) + chip_data['count']

        compute = Compute(chip_counts=chip_counts)
        total_h100e = compute.total_h100e_tpp()
        print(f"DEBUG dark_compute_dead_and_alive(year={year}): total_h100e={total_h100e}, num_chip_types={len(chip_counts)}", flush=True)
        return compute

    def annual_hazard_rate_after_years_of_life(self, years_of_life: float) -> float:
        return self.initial_hazard_rate + self.increase_in_hazard_rate_per_year * years_of_life

    def dark_compute(self, year : float):
        """Calculate total surviving compute at the given year.

        This applies survival rates to each chip type added in each year.

        Args:
            year: The year to calculate surviving compute for

        Returns:
            Compute object containing surviving chips
        """
        chip_counts = {}
        total_surviving_compute = 0
        debug_info = []

        for y in self.dark_compute_added_per_year:
            years_of_life = year - y
            # Skip if compute hasn't been added yet (negative years of life)
            if years_of_life < 0:
                continue

            # Calculate survival rate
            cumulative_hazard = self.initial_hazard_rate * years_of_life + self.increase_in_hazard_rate_per_year * years_of_life**2 / 2
            survival_rate = np.exp(-cumulative_hazard)

            # Apply survival rate to all chips added in year y
            year_total_compute = 0.0
            for chip_id, chip_data in self.dark_compute_added_per_year[y].items():
                # Calculate per-chip values
                h100e_tpp_per_chip = chip_data['h100_equivalence'] / chip_data['count'] if chip_data['count'] > 0 else 0

                # Calculate energy per chip
                tpp_per_chip = h100e_tpp_per_chip * H100_TPP_PER_CHIP
                watts_per_chip = tpp_per_chip * H100_WATTS_PER_TPP / chip_data['energy_efficiency_relative_to_h100']

                # Create Chip object
                chip = Chip(
                    h100e_tpp_per_chip=h100e_tpp_per_chip,
                    W_of_energy_consumed=watts_per_chip,
                    inter_chip_memory_bandwidth_tbps=chip_data['bandwidth']
                )

                # Apply survival rate to chip count
                surviving_count = chip_data['count'] * survival_rate
                chip_counts[chip] = chip_counts.get(chip, 0) + surviving_count

                chip_compute = chip_data['h100_equivalence'] * chip_data['count']
                year_total_compute += chip_compute

            surviving_compute_from_year_y = year_total_compute * survival_rate
            total_surviving_compute += surviving_compute_from_year_y

            if len(debug_info) < 3:  # Only log first 3 entries
                debug_info.append(f"y={y}, years_of_life={years_of_life:.2f}, cumulative_hazard={cumulative_hazard:.4f}, survival_rate={survival_rate:.4f}, surviving_compute={surviving_compute_from_year_y:.2f}, compute_added={year_total_compute:.2f}")

        compute = Compute(chip_counts=chip_counts)
        if debug_info:
            print(f"DEBUG dark_compute(year={year}): total_surviving={total_surviving_compute:.4f}, num_chip_types={len(chip_counts)}, initial_hazard={self.initial_hazard_rate}, increase_per_year={self.increase_in_hazard_rate_per_year:.4f}", flush=True)
            for info in debug_info:
                print(f"  {info}", flush=True)
        return compute

    def dark_compute_energy_by_source(self, year: float) -> tuple[float, float, float, float]:
        """Calculate energy consumption and compute separated by source (initial stock vs fab-produced).

        Args:
            year: The year to calculate energy for

        Returns:
            Tuple of (initial_stock_energy_gw, fab_energy_gw, initial_h100e_tpp, fab_h100e_tpp)
        """
        initial_energy = 0.0
        fab_energy = 0.0
        initial_h100e = 0.0
        fab_h100e = 0.0

        for y in self.dark_compute_added_per_year:
            years_of_life = year - y
            if years_of_life < 0:
                continue

            # Calculate survival rate
            cumulative_hazard = self.initial_hazard_rate * years_of_life + self.increase_in_hazard_rate_per_year * years_of_life**2 / 2
            survival_rate = np.exp(-cumulative_hazard)

            # Process all chips added in year y
            for chip_id, chip_data in self.dark_compute_added_per_year[y].items():
                # Calculate per-chip values
                h100e_tpp_per_chip = chip_data['h100_equivalence'] / chip_data['count'] if chip_data['count'] > 0 else 0

                # Calculate energy per chip
                tpp_per_chip = h100e_tpp_per_chip * H100_TPP_PER_CHIP
                watts_per_chip = tpp_per_chip * H100_WATTS_PER_TPP / chip_data['energy_efficiency_relative_to_h100']

                # Calculate surviving energy and compute
                surviving_count = chip_data['count'] * survival_rate
                energy_gw = (watts_per_chip * surviving_count) / 1e9
                h100e_tpp = h100e_tpp_per_chip * surviving_count

                # Classify by source
                if chip_id == 'initial_prc_stock':
                    initial_energy += energy_gw
                    initial_h100e += h100e_tpp
                else:
                    fab_energy += energy_gw
                    fab_h100e += h100e_tpp

        return (initial_energy, fab_energy, initial_h100e, fab_h100e)

    def operational_dark_compute(self, year: float, datacenter_capacity_gw: float):
        """Calculate operational dark compute limited by datacenter energy capacity.

        This gets all surviving dark compute and scales it down if the energy requirements
        exceed the available datacenter capacity.

        Args:
            year: The year to calculate operational compute for
            datacenter_capacity_gw: Available datacenter energy capacity in GW

        Returns:
            Compute object containing chips that can be powered with available capacity
        """
        # Get all surviving dark compute
        all_dark_compute = self.dark_compute(year)

        # Calculate total energy requirements
        total_energy_required_gw = all_dark_compute.total_energy_requirements_GW()

        # If energy requirements are within capacity, return all dark compute
        if total_energy_required_gw <= datacenter_capacity_gw:
            return all_dark_compute

        # Otherwise, scale down chip counts proportionally to fit within capacity
        scaling_factor = datacenter_capacity_gw / total_energy_required_gw

        # Create new chip_counts with scaled values
        operational_chip_counts = {
            chip: count * scaling_factor
            for chip, count in all_dark_compute.chip_counts.items()
        }

        operational_compute = Compute(chip_counts=operational_chip_counts)

        print(f"DEBUG operational_dark_compute(year={year}): capacity={datacenter_capacity_gw:.2f}GW, "
              f"required={total_energy_required_gw:.2f}GW, scaling_factor={scaling_factor:.4f}, "
              f"operational_tpp={operational_compute.total_h100e_tpp():.2f}", flush=True)

        return operational_compute