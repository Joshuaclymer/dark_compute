from typing import Dict
from black_project_backend.util import lr_over_time_vs_num_workers


class ResearcherHeadcount:
    """
    Tracks researcher headcount over time and calculates detection likelihood ratios.

    The likelihood ratio is based on the idea that a larger researcher headcount
    increases the probability of detection over time.
    """

    def __init__(
        self,
        researcher_headcount_by_year: Dict[float, int],
        mean_detection_time_100_researchers: float,
        mean_detection_time_1000_researchers: float,
        variance_of_detection_time: float
    ):
        """
        Initialize ResearcherHeadcount tracker.

        Args:
            researcher_headcount_by_year: Dict mapping years to number of researchers
            mean_detection_time_100_researchers: Mean time to detection with 100 researchers
            mean_detection_time_1000_researchers: Mean time to detection with 1000 researchers
            variance_of_detection_time: Variance parameter for detection time distribution
        """
        self.researcher_headcount_by_year = researcher_headcount_by_year
        self.mean_detection_time_100_researchers = mean_detection_time_100_researchers
        self.mean_detection_time_1000_researchers = mean_detection_time_1000_researchers
        self.variance_of_detection_time = variance_of_detection_time

        # Cache for likelihood ratios
        self._lr_by_year: Dict[float, float] = None

    @property
    def researcher_headcount(self) -> Dict[float, int]:
        """Return the researcher headcount by year."""
        return self.researcher_headcount_by_year

    def _compute_lr_by_year(self) -> Dict[float, float]:
        """Compute and cache likelihood ratios for all years."""
        if self._lr_by_year is None:
            self._lr_by_year = lr_over_time_vs_num_workers(
                labor_by_year=self.researcher_headcount_by_year,
                mean_detection_time_100_workers=self.mean_detection_time_100_researchers,
                mean_detection_time_1000_workers=self.mean_detection_time_1000_researchers,
                variance_theta=self.variance_of_detection_time
            )
        return self._lr_by_year

    def get_cumulative_likelihood_ratio(self, year: float) -> float:
        """
        Get the cumulative likelihood ratio at a given year.

        The likelihood ratio represents the evidence for/against the existence
        of the researcher group based on detection signals up to this year.

        Args:
            year: The year to get the likelihood ratio for

        Returns:
            The cumulative likelihood ratio at the given year.
            - LR > 1: Evidence for existence (more likely detected if exists)
            - LR < 1: Evidence against existence (less likely detected if exists)
            - LR = 1: No evidence either way
        """
        lr_by_year = self._compute_lr_by_year()

        if not lr_by_year:
            return 1.0

        # Find the closest year that is <= the requested year
        available_years = sorted(lr_by_year.keys())

        closest_year = None
        for y in available_years:
            if y <= year:
                closest_year = y
            else:
                break

        if closest_year is None:
            return 1.0

        return lr_by_year[closest_year]
