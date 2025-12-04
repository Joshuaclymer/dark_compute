from dataclasses import dataclass
from typing import Tuple

from classes.negotiator import Negotiator


@dataclass
class ComputeAndResearchCaps:
    """Caps on compute and research imposed by the AI agreement."""
    compute_cap: float  # Maximum allowed compute (in H100e)
    research_cap: float  # Maximum allowed research effort


class AIAgreementTerms:
    """
    Represents the terms of an AI agreement, including compute and research caps.

    The agreement terms determine what compute and research levels are feasible
    based on a negotiator's beliefs about covert projects.
    """

    def __init__(self):
        pass

    def get_feasible_compute_and_research_caps_for_negotiator(self, negotiator: Negotiator) -> ComputeAndResearchCaps:
        """
        Calculate feasible compute and research caps based on negotiator's beliefs.

        Args:
            negotiator: The negotiator whose beliefs determine the feasible caps

        Returns:
            ComputeAndResearchCaps containing the maximum allowed compute and research
        """
        # TODO: Implement the logic for determining feasible caps based on negotiator beliefs
        # For now, return placeholder values
        return ComputeAndResearchCaps(
            compute_cap=0.0,
            research_cap=0.0
        )
    

