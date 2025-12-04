"""
LargestAIProject class for tracking the compute stock and energy efficiency
of the largest AI project over time.
"""

from black_project_backend.black_project_parameters import ExogenousTrends


class LargestAIProject:
    """
    Tracks the compute stock and energy efficiency of the largest AI project.

    The largest AI project's compute grows exponentially from a base value in 2025.
    Energy efficiency improves over time relative to the H100 baseline.
    """

    def __init__(self, exogenous_trends: ExogenousTrends):
        """
        Initialize with exogenous trends parameters.

        Args:
            exogenous_trends: ExogenousTrends dataclass containing:
                - largest_ai_project_compute_stock_in_2025: Initial compute in H100e
                - annual_growth_rate_of_largest_ai_project_compute_stock: Growth multiplier per year
                - improvement_in_energy_efficiency_per_year: Energy efficiency improvement per year
        """
        self.compute_stock_2025 = exogenous_trends.largest_ai_project_compute_stock_in_2025
        self.annual_growth_rate = exogenous_trends.annual_growth_rate_of_largest_ai_project_compute_stock
        self.energy_efficiency_improvement_per_year = exogenous_trends.improvement_in_energy_efficiency_per_year

    def get_compute_stock(self, year: float) -> float:
        """
        Get the compute stock of the largest AI project at a given year.

        Compute grows exponentially from the 2025 baseline.

        Args:
            year: The year to calculate compute for

        Returns:
            Compute stock in H100 equivalents
        """
        years_since_2025 = year - 2025
        return self.compute_stock_2025 * (self.annual_growth_rate ** years_since_2025)

    def get_compute_trajectory(self, years: list[float]) -> list[float]:
        """
        Get the compute stock trajectory for a list of years.

        Args:
            years: List of years to calculate compute for

        Returns:
            List of compute stock values in H100 equivalents
        """
        return [self.get_compute_stock(year) for year in years]

    def get_energy_efficiency_relative_to_h100(self, year: float) -> float:
        """
        Get the energy efficiency of the largest AI project relative to H100 at a given year.

        Energy efficiency improves over time. In 2022 (H100 release), efficiency is 1.0.
        Each year after, efficiency improves by the improvement_in_energy_efficiency_per_year factor.

        Args:
            year: The year to calculate efficiency for

        Returns:
            Energy efficiency relative to H100 (>1 means more efficient than H100)
        """
        # H100 was released in 2022, so that's our baseline year
        years_since_h100 = year - 2022
        if years_since_h100 < 0:
            # Before H100, assume same efficiency as H100 (conservative)
            return 1.0
        return self.energy_efficiency_improvement_per_year ** years_since_h100

    def get_energy_efficiency_trajectory(self, years: list[float]) -> list[float]:
        """
        Get the energy efficiency trajectory for a list of years.

        Args:
            years: List of years to calculate efficiency for

        Returns:
            List of energy efficiency values relative to H100
        """
        return [self.get_energy_efficiency_relative_to_h100(year) for year in years]
