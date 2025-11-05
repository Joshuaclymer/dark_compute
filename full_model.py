from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

"""

UPDATED FULL MODEL: Covert Compute Production Forecasting Model with Multi-Channel Detection
============================================================================================

This model estimates compute production and detection risk for a covert semiconductor
fab project using a multi-channel detection probability framework.

MODEL OVERVIEW:
--------------
The key change from the original model is the detection probability calculation.
Instead of a single detection mechanism based on worker count, we now model three
independent detection channels:

P(detection) = 1 - [1 - P(detection | inventory accounting)]
                 × [1 - P(detection | procurement accounting)]
                 × [1 - P(detection | other)]

DETECTION CHANNELS:
------------------

1. INVENTORY ACCOUNTING:
   - Based on US intelligence estimates of PRC's semiconductor manufacturing equipment (SME)
   - Strategy: Estimate total SME the PRC has, track how much goes "missing"
   - Uses empirical data from diversion_vs_detection.py in intelligence_error_bars/
   - Curve shows: % SME diverted → P(diversion outside intelligence error bars)
   - The fitted curve is: P(detection) = 1 - exp(-k * percent_diverted)
     where k ≈ 0.05 (fitted from historical nuclear program intelligence data)
   - To apply this:
     * Calculate how much SME is needed for the covert fab
     * Estimate what % of total PRC SME this represents
     * Use the curve to find P(detection | inventory accounting)

2. PROCUREMENT ACCOUNTING:
   - [To be implemented]
   - Would track unusual procurement patterns, supply chain anomalies

3. OTHER DETECTION CHANNELS:
   - [To be implemented]
   - Could include HUMINT, SIGINT, satellite imagery, etc.

CURRENT IMPLEMENTATION STATUS:
-----------------------------
This initial version implements ONLY the inventory accounting channel.
The other channels are set to 0 for now and will be added in subsequent updates.

MODEL INPUTS:
-------------
Time inputs:
1. agreement_start_year: The year the covert agreement/construction begins
2. year: The year for which to determine the cumulative detection probability and compute production rate

Strategic choice inputs (chosen by PRC):
3. construction_labor: Number of people concurrently involved in constructing the fab. 
4. operating_labor: Number of people concurrently involved in operating the fab 
5. process_node: The process node (nm) that the covert fab will use
6z. proportion_of_sme_diverted: What proportion (0-1) of total PRC SME is diverted to covert fab

MODEL OUTPUTS:
--------------
1. cumulative_detection_probability: Probability that the project has been detected by input "year"
2. annual_compute_production: Distribution of H100-equivalent compute production during "year"

DETECTION CALCULATION DETAILS:
------------------------------
For inventory accounting:
- The PRC chooses what proportion of their total SME to divert (strategic input)
- Convert proportion to percentage: percent_diverted = proportion × 100
- Use the empirical curve: P(detection | inventory) = 1 - exp(-k * percent_diverted)
  where k ≈ 0.05 (fitted from historical intelligence data)
- This detection probability is time-independent in the current model
  (represents the probability per unit time that the diversion is noticed)
- The cumulative probability by year Y would integrate this over time from agreement start

The curve is derived from historical data comparing:
- Intelligence estimates vs ground truth for covert nuclear programs
- Shows the probability that a given % error/diversion exceeds intelligence error bars
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================================
# CLASSES 
# ============================================================================

@dataclass
class ProcessNode(Enum):
    nm130 = "130nm"
    nm28 = "28nm"
    nm14 = "14nm"
    nm7 = "7nm"

# ============================================================================
# SIMULATION INPUTS (these are supplied at the start of a world simulation)
# ============================================================================

@dataclass
class PRCStrategy:
    # Covert fab
    probability_prc_builds_a_covert_fab : float
    when_prc_starts_building_a_covert_fab : Optional[float]
    covert_fab_operating_headcount : Optional[int]
    covert_fab_construction_headcount : Optional[int]
    covert_fab_process_node : Optional[ProcessNode]
    proportion_of_indigenous_PRC_SME_diverted_to_covert_fab : Optional[float]

@dataclass
class USStrategy:
    us_strategy : float # just leave this incomplete for now

@dataclass
class Inputs:
    year : float 
    us_prc_agreement_in_force_year : float
    prc_strategy = PRCStrategy
    us_strategy = USStrategy

# ============================================================================
# SIMULATION STATE
# ============================================================================

@dataclass
class SimulationState:
    US_intelligence_estimate_of_probability_PRC_covert_project_exists : float
    h100e_of_PRC_covert_project : float

# ============================================================================
# World Properties (properties of the world for a specific rollout)
# ============================================================================


def p_detection_via_procurement_accounting(what_node_to_use, nodes_with_sufficient_localization):
    return what_node_to_use not in nodes_with_sufficient_localization

def p_detection_via_other_strategies(construction_labor, construction_duration, )


@dataclass
class WorldProperties:
    # PRC properties
    PRC_SME_stock : float
    PRC_chip_efficiency_relative_to_h100 : float
    process_nodes_with_90p_PRC_localization : List[ProcessNode]
    PRC_covert_fab : PRCCovertFab
    
    # PRC covert project 
    flow_to_PRC_covert_project_from_covert_fab = float
    probability_PRC_covert_fab_is_detected = float

    def __init__(modelParameters : ModelParameters):
        self.construction_duration


 
# ============================================================================
# FAB MODEL PARAMETERS 
# ============================================================================

@dataclass
class Fab:
    # This is the proportion of indigenous SME that the PRC might divert to a covert fab such that the probability that the diversion is 'detected' by U.S. intelligence is 50%. 
    # This figure is used to fit a smooth constrained logistic curve mapping the proportion of SME diverted to the probability of detection given by: f(x) = 1 - exp(-k * percent_error)
    PROPORTION_OF_DIVERTED_SME_WITH_50p_CHANCE_OF_DETECTION = 0.14

    # These are anchors for the probabilities that the PRC will be able to source >90% of SME at a given node from domestic companies. I'm assuming 90% localization is considered sufficient to evade detection. 
    # p_sufficient_localization_of_130nm_by_2025 = TODO
    # p_sufficient_localization_of_130nm_by_2028 = TODO

# ============================================================================
# Covert fab utilities and parameters
# ============================================================================


class FabParameters:
    H100_sized_dies_per_wafer : float = 28
    

class CovertFab:

    # Inputs
    construction_start_year : float
    number_of_lithography_scanners : int
    construction_headcount : int
    process_node : str

    # Properties estimated from historical data
    construction_duration : float
    wafer_starts_per_month : float

    def __init__(
        construction_start_year : float,
        number_of_lithography_scanners : int,
        construction_headcount : int,
        process_node : str,
        yield_rate : float
    ):


    def get_H100e_per_year()

# ============================================================================
# WORLD PROPERTIES
# ============================================================================

@dataclass
class World:
    # World properties
    stock_of_PRC_SME : float
    year_when_PRC_reaches_90p_localization_at_node : dict = {"130nm": None, "28nm": None, "14nm": None, "7nm": None}
    chip_efficiency_relative_to_2022 : float

    def __init__(modelParameters : ModelParameters):
        self.construction_duration

# ============================================================================
# STRATEGIC CHOICES 
# ============================================================================


@dataclass
class CovertFabProperties:
    covert_fab_production_capacity : float
    covert_fab_process_node : float
    covert_fab_construction_duration : float
    
# ============================================================================
# DETECTION PROBABILITY UTILITIES
# ============================================================================

def p_detection_via_inventory_accounting(proportion_of_SME_diverted):
    return 1 - np.exp(-0.05 * proportion_of_SME_diverted)

def p_detection_via_procurement_accounting(what_node_to_use, nodes_with_sufficient_localization):
    return what_node_to_use not in nodes_with_sufficient_localization

def p_detection_via_other_strategies(construction_labor, construction_duration, )