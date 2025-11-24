import numpy as np
import random
from typing import List, Dict
from abc import ABC, abstractmethod
from backend.paramaters import InitialPRCDarkComputeParameters, SurvivalRateParameters

# Constants for H100 chip (duplicated from covert_fab to avoid circular import)
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per H100 chip (134 TFLOP/s FP16 * 16 bits)
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass (default, can be overridden)


def sample_prc_growth_rate(params: InitialPRCDarkComputeParameters):
    """Sample the PRC compute stock growth rate using a metalog distribution with specified percentiles."""
    from backend.util import sample_from_metalog_3term_semi_bounded_custom_percentiles
    return sample_from_metalog_3term_semi_bounded_custom_percentiles(
        params.annual_growth_rate_of_prc_compute_stock_p10,
        params.annual_growth_rate_of_prc_compute_stock_p50,
        params.annual_growth_rate_of_prc_compute_stock_p90
    )

def compute_prc_compute_stock(year, growth_rate, params: InitialPRCDarkComputeParameters):
    """Calculate PRC compute stock for a given year using a specific growth rate"""
    years_since_2025 = year - 2025
    return params.total_prc_compute_stock_in_2025 * (growth_rate ** years_since_2025)

def sample_us_estimate_of_prc_compute_stock(prc_compute_stock, params: InitialPRCDarkComputeParameters):
    # Calculate pdf of absolute relative error
    k = -np.log(0.5) / params.us_intelligence_median_error_in_estimate_of_prc_compute_stock

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


def lr_from_prc_compute_accounting(reported_prc_compute_stock, optimal_diversion_proportion, us_estimate_of_prc_compute_stock, params: InitialPRCDarkComputeParameters):
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
    k = -np.log(0.5) / params.us_intelligence_median_error_in_estimate_of_prc_compute_stock
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


def sample_hazard_rate_multiplier(params: SurvivalRateParameters) -> float:
    """Sample a multiplier for hazard rates from a metalog distribution.

    Returns a multiplier that will be applied to both initial_hazard_rate and
    increase_of_hazard_rate_per_year to create correlated uncertainty.
    """
    from backend.util import sample_from_metalog_3term_semi_bounded

    # Compute absolute percentiles from the ratios
    p25 = params.hazard_rate_p25_relative_to_p50
    p50 = 1.0  # Median multiplier is 1.0
    p75 = params.hazard_rate_p75_relative_to_p50

    # Sample multiplier from semi-bounded metalog distribution (lower bound 0, no upper bound)
    return sample_from_metalog_3term_semi_bounded(p25, p50, p75)

def sample_hazard_rates(params: SurvivalRateParameters) -> tuple[float, float]:
    """Sample both initial hazard rate and increase rate using a common multiplier.

    Returns:
        tuple: (initial_hazard_rate, increase_of_hazard_rate_per_year)
    """
    multiplier = sample_hazard_rate_multiplier(params)

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

    def __init__(self, agreement_year, proportion_of_initial_compute_stock_to_divert, optimal_proportion_of_initial_compute_stock_to_divert, initial_compute_parameters: InitialPRCDarkComputeParameters, survival_parameters: SurvivalRateParameters):
        self.agreement_year = agreement_year
        self.initial_compute_parameters = initial_compute_parameters
        assert proportion_of_initial_compute_stock_to_divert < self.initial_compute_parameters.proportion_of_prc_chip_stock_produced_domestically, "This model assumes that the PRC only diverts domestically-produced chips to a covert project. Therefore proportion to divert < proportion produced domestically."
        self.survival_parameters = survival_parameters

        # Sample growth rate once for this simulation
        self.prc_growth_rate = sample_prc_growth_rate(initial_compute_parameters)

        # Calculate initial PRC stock using the sampled growth rate
        self.initial_prc_stock = compute_prc_compute_stock(agreement_year, self.prc_growth_rate, initial_compute_parameters)
        self.initial_prc_dark_compute = self.initial_prc_stock * proportion_of_initial_compute_stock_to_divert

        # dark_compute_added_per_year now stores chip dictionaries instead of single numbers
        # Structure: {year: {chip_id: {'count': float, 'h100_equivalence': float,
        #                                'energy_efficiency_relative_to_h100': float, 'bandwidth': float}}}
        self.dark_compute_added_per_year = {}

        # Initialize with a default chip representing the initial PRC dark compute stock
        self.dark_compute_added_per_year[agreement_year] = {
            'initial_prc_stock': {
                'count': 1.0,  # Normalized count
                'h100_equivalence': self.initial_prc_dark_compute,
                'energy_efficiency_relative_to_h100': initial_compute_parameters.energy_efficiency_relative_to_h100,
                'bandwidth': 1.8  # Default inter-chip bandwidth in tbps (H100-like)
            }
        }

        self.us_estimate_of_prc_stock = sample_us_estimate_of_prc_compute_stock(self.initial_prc_stock, initial_compute_parameters)
        self.lr_from_prc_compute_accounting = lr_from_prc_compute_accounting(
            reported_prc_compute_stock=self.initial_prc_stock - self.initial_prc_dark_compute,
            optimal_diversion_proportion=optimal_proportion_of_initial_compute_stock_to_divert,
            us_estimate_of_prc_compute_stock=self.us_estimate_of_prc_stock,
            params=initial_compute_parameters
        )
        self.initial_hazard_rate, self.increase_in_hazard_rate_per_year = sample_hazard_rates(survival_parameters)
    
    def add_dark_compute(self, year : float, compute_to_add):
        """Add dark compute for a year from a Compute object.

        Args:
            year: The year to add compute for
            compute_to_add: Either a Compute object with chip specifications, or a float for H100e TPP (uses defaults)
        """
        if year not in self.dark_compute_added_per_year:
            self.dark_compute_added_per_year[year] = {}

        # Handle both Compute objects and scalar values
        if isinstance(compute_to_add, Compute):
            # Extract chip characteristics from the Compute object
            chip_id_counter = len(self.dark_compute_added_per_year[year])

            for chip, count in compute_to_add.chip_counts.items():
                # Calculate energy efficiency from the chip's properties
                # energy_efficiency_relative_to_h100 = (h100e_tpp_per_chip * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP) / W_of_energy_consumed
                energy_efficiency_relative_to_h100 = (chip.h100e_tpp_per_chip * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP) / chip.W_of_energy_consumed

                chip_id = f'added_year_{year}_chip_{chip_id_counter}'
                self.dark_compute_added_per_year[year][chip_id] = {
                    'count': count,
                    'h100_equivalence': chip.h100e_tpp_per_chip * count,
                    'energy_efficiency_relative_to_h100': energy_efficiency_relative_to_h100,
                    'bandwidth': chip.inter_chip_memory_bandwidth_tbps
                }
                chip_id_counter += 1
        else:
            # Backward compatibility: treat as scalar H100e TPP with default characteristics
            chip_id = f'added_year_{year}'
            self.dark_compute_added_per_year[year][chip_id] = {
                'count': 1.0,
                'h100_equivalence': compute_to_add,
                'energy_efficiency_relative_to_h100': self.initial_compute_parameters.energy_efficiency_relative_to_h100,
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

    def combined_likelihood_ratio(self) -> float:
        """Calculate likelihood ratio from PRC compute accounting.

        Returns:
            float: Likelihood ratio from PRC compute stock accounting
        """
        return self.lr_from_prc_compute_accounting