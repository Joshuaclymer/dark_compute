from flask import Flask, render_template, jsonify, request, send_file
from model import Model, CovertProjectStrategy, default_prc_covert_project_strategy, best_prc_covert_project_strategy
from fab_model import ProcessNode, FabModelParameters
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure likelihood ratio thresholds for detection plots
# Update this list to change all plots automatically
LIKELIHOOD_RATIOS = [1, 3, 5]

@app.route('/')
def index():
    # Pass default strategy values to template
    from stock_model import InitialPRCComputeStockParameters, SurvivalRateParameters
    from datacenter_model import CovertDatacenterParameters
    defaults = {
        'proportion_of_initial_chip_stock_to_divert': default_prc_covert_project_strategy.proportion_of_initial_compute_stock_to_divert,
        'GW_per_initial_datacenter': default_prc_covert_project_strategy.GW_per_initial_datacenter,
        'number_of_initial_datacenters': default_prc_covert_project_strategy.number_of_initial_datacenters,
        'GW_per_year_of_concealed_datacenters': default_prc_covert_project_strategy.GW_per_year_of_concealed_datacenters,
        'operating_labor': default_prc_covert_project_strategy.covert_fab_operating_labor,
        'construction_labor': default_prc_covert_project_strategy.covert_fab_construction_labor,
        'scanner_proportion': default_prc_covert_project_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted,
        'total_prc_compute_stock_in_2025': InitialPRCComputeStockParameters.total_prc_compute_stock_in_2025,
        'annual_growth_rate_of_prc_compute_stock': InitialPRCComputeStockParameters.annual_growth_rate_of_prc_compute_stock,
        'relative_sigma_of_prc_compute_stock': InitialPRCComputeStockParameters.relative_sigma_of_prc_compute_stock,
        'us_intelligence_median_error_in_estimate_of_prc_compute_stock': InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock,
        'energy_efficiency_relative_to_h100': InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100,
        'total_global_compute_in_2025': InitialPRCComputeStockParameters.total_global_compute_in_2025,
        'annual_growth_rate_of_global_compute': InitialPRCComputeStockParameters.annual_growth_rate_of_global_compute,
        'relative_sigma_of_global_compute': InitialPRCComputeStockParameters.relative_sigma_of_global_compute,
        'median_unreported_compute_owned_by_non_prc_actors': InitialPRCComputeStockParameters.median_unreported_compute_owned_by_non_prc_actors,
        'relative_sigma_unreported_compute_owned_by_non_prc_actors': InitialPRCComputeStockParameters.relative_sigma_unreported_compute_owned_by_non_prc_actors,
        'initial_hazard_rate_p50': SurvivalRateParameters.initial_hazard_rate_p50,
        'increase_of_hazard_rate_per_year_p50': SurvivalRateParameters.increase_of_hazard_rate_per_year_p50,
        'hazard_rate_p25_relative_to_p50': SurvivalRateParameters.hazard_rate_p25_relative_to_p50,
        'hazard_rate_p75_relative_to_p50': SurvivalRateParameters.hazard_rate_p75_relative_to_p50,
        'max_proportion_of_PRC_energy_consumption': CovertDatacenterParameters.max_proportion_of_PRC_energy_consumption,
        'total_GW_of_PRC_energy_consumption': CovertDatacenterParameters.total_GW_of_PRC_energy_consumption,
        'construction_labor_per_MW_per_year': CovertDatacenterParameters.construction_labor_per_MW_per_year,
        'relative_sigma_construction_labor_per_MW_per_year': CovertDatacenterParameters.relative_sigma_construction_labor_per_MW_per_year,
        'operating_labor_per_MW': CovertDatacenterParameters.operating_labor_per_MW,
        'relative_sigma_operating_labor_per_MW': CovertDatacenterParameters.relative_sigma_operating_labor_per_MW,
        'mean_detection_time_of_covert_site_for_100_workers': CovertDatacenterParameters.mean_detection_time_of_covert_site_for_100_workers,
        'mean_detection_time_of_covert_site_for_1000_workers': CovertDatacenterParameters.mean_detection_time_of_covert_site_for_1000_workers,
        'variance_of_detection_time_given_num_workers': CovertDatacenterParameters.variance_of_detection_time_given_num_workers,
    }
    return render_template('index.html', defaults=defaults)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
    except Exception as e:
        print(f"ERROR parsing request: {e}", flush=True)
        return jsonify({"error": str(e)}), 400

    # Extract parameters
    agreement_year = float(data.get('agreement_year', 2030))
    end_year = float(data.get('end_year', 2037))
    increment = float(data.get('increment', 0.1))
    num_simulations = int(data.get('num_simulations', 100))

    # PRC strategy parameters
    run_covert_project = data.get('run_covert_project', True)
    proportion_of_initial_chip_stock_to_divert = float(data.get('proportion_of_initial_chip_stock_to_divert', 0.05))

    # Datacenter strategic choices
    GW_per_initial_datacenter = float(data.get('GW_per_initial_datacenter', 5))
    number_of_initial_datacenters = float(data.get('number_of_initial_datacenters', 0.1))
    GW_per_year_of_concealed_datacenters = float(data.get('GW_per_year_of_concealed_datacenters', 1))

    # Fab strategic choices
    build_covert_fab = data.get('build_covert_fab', True)
    operating_labor = int(data.get('operating_labor', 728))
    construction_labor = int(data.get('construction_labor', 448))
    process_node_str = data.get('process_node', 'best_available_indigenously')
    scanner_proportion = float(data.get('scanner_proportion', 0.102))

    print(f"DEBUG: Received scanner_proportion = {scanner_proportion}", flush=True)
    print(f"DEBUG: operating_labor = {operating_labor}, construction_labor = {construction_labor}", flush=True)

    # US prior probabilities
    p_project_exists = float(data.get('p_project_exists', 0.2))
    p_fab_exists = float(data.get('p_fab_exists', 0.1))

    # Initial PRC compute stock parameters
    from stock_model import InitialPRCComputeStockParameters
    print(f"DEBUG: Received compute stock params from client: total_2025={data.get('total_prc_compute_stock_in_2025')}, growth={data.get('annual_growth_rate_of_prc_compute_stock')}, sigma={data.get('relative_sigma_of_prc_compute_stock')}", flush=True)
    if 'total_prc_compute_stock_in_2025' in data:
        InitialPRCComputeStockParameters.total_prc_compute_stock_in_2025 = float(data['total_prc_compute_stock_in_2025'])
    if 'annual_growth_rate_of_prc_compute_stock' in data:
        InitialPRCComputeStockParameters.annual_growth_rate_of_prc_compute_stock = float(data['annual_growth_rate_of_prc_compute_stock'])
    if 'relative_sigma_of_prc_compute_stock' in data:
        InitialPRCComputeStockParameters.relative_sigma_of_prc_compute_stock = float(data['relative_sigma_of_prc_compute_stock'])
    if 'us_intelligence_median_error_in_estimate_of_prc_compute_stock' in data:
        InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock = float(data['us_intelligence_median_error_in_estimate_of_prc_compute_stock'])
    if 'energy_efficiency_relative_to_h100' in data:
        InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100 = float(data['energy_efficiency_relative_to_h100'])
    print(f"DEBUG: Updated params to: total_2025={InitialPRCComputeStockParameters.total_prc_compute_stock_in_2025}, growth={InitialPRCComputeStockParameters.annual_growth_rate_of_prc_compute_stock}, sigma={InitialPRCComputeStockParameters.relative_sigma_of_prc_compute_stock}", flush=True)

    # Survival rate parameters
    from stock_model import SurvivalRateParameters
    from util import clear_metalog_cache

    survival_params_changed = False
    if 'initial_hazard_rate_p50' in data:
        SurvivalRateParameters.initial_hazard_rate_p50 = float(data['initial_hazard_rate_p50'])
        survival_params_changed = True
    if 'increase_of_hazard_rate_per_year_p50' in data:
        SurvivalRateParameters.increase_of_hazard_rate_per_year_p50 = float(data['increase_of_hazard_rate_per_year_p50'])
        survival_params_changed = True
    if 'hazard_rate_p25_relative_to_p50' in data:
        SurvivalRateParameters.hazard_rate_p25_relative_to_p50 = float(data['hazard_rate_p25_relative_to_p50'])
        survival_params_changed = True
    if 'hazard_rate_p75_relative_to_p50' in data:
        SurvivalRateParameters.hazard_rate_p75_relative_to_p50 = float(data['hazard_rate_p75_relative_to_p50'])
        survival_params_changed = True

    # Clear metalog cache if survival parameters changed
    if survival_params_changed:
        clear_metalog_cache()

    # Fab model parameters - update all parameters from data
    from fab_model import clear_detection_constants_cache

    detection_params_changed = False
    if 'median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock' in data:
        FabModelParameters.median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock = float(data['median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock'])
    if 'mean_detection_time_for_100_workers' in data:
        FabModelParameters.mean_detection_time_for_100_workers = float(data['mean_detection_time_for_100_workers'])
        detection_params_changed = True
    if 'mean_detection_time_for_1000_workers' in data:
        FabModelParameters.mean_detection_time_for_1000_workers = float(data['mean_detection_time_for_1000_workers'])
        detection_params_changed = True
    if 'variance_of_detection_time_given_num_workers' in data:
        FabModelParameters.variance_of_detection_time_given_num_workers = float(data['variance_of_detection_time_given_num_workers'])
        detection_params_changed = True

    # Clear detection constants cache if detection parameters changed
    if detection_params_changed:
        clear_detection_constants_cache()
    if 'wafers_per_month_per_worker' in data:
        FabModelParameters.wafers_per_month_per_worker = float(data['wafers_per_month_per_worker'])
    if 'labor_productivity_relative_sigma' in data:
        FabModelParameters.labor_productivity_relative_sigma = float(data['labor_productivity_relative_sigma'])
    if 'wafers_per_month_per_lithography_scanner' in data:
        FabModelParameters.wafers_per_month_per_lithography_scanner = float(data['wafers_per_month_per_lithography_scanner'])
    if 'scanner_productivity_relative_sigma' in data:
        FabModelParameters.scanner_productivity_relative_sigma = float(data['scanner_productivity_relative_sigma'])
    if 'construction_time_for_5k_wafers_per_month' in data:
        FabModelParameters.construction_time_for_5k_wafers_per_month = float(data['construction_time_for_5k_wafers_per_month'])
    if 'construction_time_for_100k_wafers_per_month' in data:
        FabModelParameters.construction_time_for_100k_wafers_per_month = float(data['construction_time_for_100k_wafers_per_month'])
    if 'construction_time_relative_sigma' in data:
        FabModelParameters.construction_time_relative_sigma = float(data['construction_time_relative_sigma'])
    if 'construction_workers_per_1000_wafers_per_month' in data:
        FabModelParameters.construction_workers_per_1000_wafers_per_month = float(data['construction_workers_per_1000_wafers_per_month'])
    if 'h100_sized_chips_per_wafer' in data:
        FabModelParameters.h100_sized_chips_per_wafer = float(data['h100_sized_chips_per_wafer'])
    if 'transistor_density_scaling_exponent' in data:
        FabModelParameters.transistor_density_scaling_exponent = float(data['transistor_density_scaling_exponent'])
    if 'architecture_efficiency_improvement_per_year' in data:
        FabModelParameters.architecture_efficiency_improvement_per_year = float(data['architecture_efficiency_improvement_per_year'])
    if 'prc_additional_lithography_scanners_produced_per_year' in data:
        FabModelParameters.prc_additional_lithography_scanners_produced_per_year = float(data['prc_additional_lithography_scanners_produced_per_year'])
    if 'prc_lithography_scanners_produced_in_first_year' in data:
        FabModelParameters.prc_lithography_scanners_produced_in_first_year = float(data['prc_lithography_scanners_produced_in_first_year'])
    if 'prc_scanner_production_relative_sigma' in data:
        FabModelParameters.prc_scanner_production_relative_sigma = float(data['prc_scanner_production_relative_sigma'])

    # Update localization probabilities (only two points: 2025 and 2031)
    if 'localization_130nm_2025' in data:
        FabModelParameters.Probability_of_90p_PRC_localization_at_node[ProcessNode.nm130] = [
            (2025, float(data['localization_130nm_2025'])),
            (2031, float(data['localization_130nm_2031']))
        ]
    if 'localization_28nm_2025' in data:
        FabModelParameters.Probability_of_90p_PRC_localization_at_node[ProcessNode.nm28] = [
            (2025, float(data['localization_28nm_2025'])),
            (2031, float(data['localization_28nm_2031']))
        ]
    if 'localization_14nm_2025' in data:
        FabModelParameters.Probability_of_90p_PRC_localization_at_node[ProcessNode.nm14] = [
            (2025, float(data['localization_14nm_2025'])),
            (2031, float(data['localization_14nm_2031']))
        ]
    if 'localization_7nm_2025' in data:
        FabModelParameters.Probability_of_90p_PRC_localization_at_node[ProcessNode.nm7] = [
            (2025, float(data['localization_7nm_2025'])),
            (2031, float(data['localization_7nm_2031']))
        ]

    # Covert datacenter parameters
    from datacenter_model import CovertDatacenterParameters

    if 'max_proportion_of_PRC_energy_consumption' in data:
        CovertDatacenterParameters.max_proportion_of_PRC_energy_consumption = float(data['max_proportion_of_PRC_energy_consumption'])
    if 'total_GW_of_PRC_energy_consumption' in data:
        CovertDatacenterParameters.total_GW_of_PRC_energy_consumption = float(data['total_GW_of_PRC_energy_consumption'])
    if 'construction_labor_per_MW_per_year' in data:
        CovertDatacenterParameters.construction_labor_per_MW_per_year = float(data['construction_labor_per_MW_per_year'])
    if 'relative_sigma_construction_labor_per_MW_per_year' in data:
        CovertDatacenterParameters.relative_sigma_construction_labor_per_MW_per_year = float(data['relative_sigma_construction_labor_per_MW_per_year'])
    if 'operating_labor_per_MW' in data:
        CovertDatacenterParameters.operating_labor_per_MW = float(data['operating_labor_per_MW'])
    if 'relative_sigma_operating_labor_per_MW' in data:
        CovertDatacenterParameters.relative_sigma_operating_labor_per_MW = float(data['relative_sigma_operating_labor_per_MW'])
    if 'mean_detection_time_of_covert_site_for_100_workers' in data:
        CovertDatacenterParameters.mean_detection_time_of_covert_site_for_100_workers = float(data['mean_detection_time_of_covert_site_for_100_workers'])
    if 'mean_detection_time_of_covert_site_for_1000_workers' in data:
        CovertDatacenterParameters.mean_detection_time_of_covert_site_for_1000_workers = float(data['mean_detection_time_of_covert_site_for_1000_workers'])
    if 'variance_of_detection_time_given_num_workers' in data:
        CovertDatacenterParameters.variance_of_detection_time_given_num_workers = float(data['variance_of_detection_time_given_num_workers'])

    try:
        # Create custom covert project strategy with user parameters
        # Map process node string to enum (or keep as string for "best_available_indigenously")
        process_node_map = {
            'best_available_indigenously': 'best_available_indigenously',  # Keep as string
            'nm130': ProcessNode.nm130,
            'nm28': ProcessNode.nm28,
            'nm14': ProcessNode.nm14,
            'nm7': ProcessNode.nm7
        }
        process_node = process_node_map.get(process_node_str, process_node_str)

        print(f"DEBUG: About to create strategy...", flush=True)
        custom_strategy = CovertProjectStrategy(
            run_a_covert_project=run_covert_project,
            proportion_of_initial_compute_stock_to_divert=proportion_of_initial_chip_stock_to_divert,
            GW_per_initial_datacenter=GW_per_initial_datacenter,
            number_of_initial_datacenters=number_of_initial_datacenters,
            GW_per_year_of_concealed_datacenters=GW_per_year_of_concealed_datacenters,
            build_a_covert_fab=build_covert_fab,
            covert_fab_operating_labor=operating_labor,
            covert_fab_construction_labor=construction_labor,
            covert_fab_process_node=process_node,
            covert_fab_proportion_of_prc_lithography_scanners_devoted=scanner_proportion
        )

        print(f"DEBUG: Created strategy with scanner_proportion = {custom_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted}", flush=True)
    except Exception as e:
        print(f"ERROR creating strategy: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # Create and run model
    model = Model(
        year_us_prc_agreement_goes_into_force=agreement_year,
        end_year=end_year,
        increment=increment,
        prc_strategy=custom_strategy  # PRC's actual strategy from user input
    )

    # Update initial detector beliefs (US's prior probabilities)
    # Note: US beliefs about PRC strategy are always best_prc_covert_project_strategy
    model.initial_detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"][agreement_year].p_project_exists = p_project_exists
    model.initial_detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"][agreement_year].p_covert_fab_exists = p_fab_exists

    try:
        # Run simulations
        print(f"DEBUG: Running simulations...", flush=True)
        model.run_simulations(num_simulations=num_simulations)

        # Debug: Check a sample fab's production capacity
        if model.simulation_results:
            covert_projects, _ = model.simulation_results[0]
            if 'prc_covert_project' in covert_projects:
                fab = covert_projects['prc_covert_project'].covert_fab
                if fab:
                    print(f"DEBUG: Sample fab construction_start_year = {fab.construction_start_year}", flush=True)
                    print(f"DEBUG: Sample fab construction_duration = {fab.construction_duration}", flush=True)
                    print(f"DEBUG: Sample fab total_prc_lithography_scanners_for_node = {fab.total_prc_lithography_scanners_for_node}", flush=True)
                    print(f"DEBUG: Sample fab process_node = {fab.process_node}", flush=True)
                    print(f"DEBUG: Sample fab wafer_starts_per_month = {fab.wafer_starts_per_month}", flush=True)
                    print(f"DEBUG: Sample fab production_capacity = {fab.production_capacity}", flush=True)

        # Sample initial dark compute stock distribution
        from stock_model import PRCDarkComputeStock, InitialPRCComputeStockParameters as StockParams
        print(f"DEBUG: Sampling with params - total_2025={StockParams.total_prc_compute_stock_in_2025}, growth_rate={StockParams.annual_growth_rate_of_prc_compute_stock}, sigma={StockParams.relative_sigma_of_prc_compute_stock}, divert_proportion={proportion_of_initial_chip_stock_to_divert}", flush=True)
        initial_prc_stock_samples = []  # Total PRC compute stock before diversion
        initial_compute_samples = []  # Dark compute diverted to covert project
        initial_lr_samples = []  # Combined likelihood ratios for detection
        lr_prc_accounting_samples = []  # Individual LR from PRC compute accounting
        lr_global_accounting_samples = []  # Individual LR from global compute production accounting
        for _ in range(1000):
            compute_stock = PRCDarkComputeStock(agreement_year, proportion_of_initial_chip_stock_to_divert, best_prc_covert_project_strategy.proportion_of_initial_compute_stock_to_divert)
            initial_prc_stock_samples.append(compute_stock.initial_prc_stock)
            initial_compute_samples.append(compute_stock.initial_prc_dark_compute)
            # Get individual LRs
            lr_prc = compute_stock.lr_from_prc_compute_accounting
            lr_global = compute_stock.lr_from_global_compute_production_accounting
            lr_prc_accounting_samples.append(lr_prc)
            lr_global_accounting_samples.append(lr_global)
            # Get combined LR from both PRC and global compute accounting
            combined_lr = lr_prc * lr_global
            initial_lr_samples.append(combined_lr)
        print(f"DEBUG: Sampled {len(initial_compute_samples)} dark compute samples, min={min(initial_compute_samples):.0f}, max={max(initial_compute_samples):.0f}, mean={sum(initial_compute_samples)/len(initial_compute_samples):.0f}", flush=True)

        # Calculate detection probabilities for different LR thresholds
        detection_probs = {}
        for threshold in LIKELIHOOD_RATIOS:
            num_detected = sum(1 for lr in initial_lr_samples if lr >= threshold)
            detection_probs[f"{threshold}x"] = num_detected / len(initial_lr_samples)
        print(f"DEBUG: Detection probabilities - {detection_probs}", flush=True)

        # Extract data for plots
        print(f"DEBUG: Extracting plot data...", flush=True)
        results = extract_plot_data(model, p_fab_exists, p_project_exists)

        # Add initial compute stock distribution to results
        results['initial_prc_stock_samples'] = initial_prc_stock_samples
        results['initial_compute_stock_samples'] = initial_compute_samples
        results['initial_dark_compute_detection_probs'] = detection_probs
        results['lr_prc_accounting_samples'] = lr_prc_accounting_samples
        results['lr_global_accounting_samples'] = lr_global_accounting_samples
        results['lr_combined_samples'] = initial_lr_samples
        results['diversion_proportion'] = proportion_of_initial_chip_stock_to_divert

        print(f"DEBUG: Returning results...", flush=True)
        return jsonify(results)
    except Exception as e:
        print(f"ERROR running simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def extract_plot_data(model, p_fab_exists, p_project_exists):
    """Extract plot data from model simulation results"""
    from stock_model import H100_TPP_PER_CHIP, H100_WATTS_PER_TPP

    if not model.simulation_results:
        return {"error": "No simulation results"}

    agreement_year = model.year_us_prc_agreement_goes_into_force

    # Extract time series data (like simulation_runs.png)
    all_years = []
    us_probs_by_sim = []
    h100e_by_sim = []

    # Individual fab detection likelihood ratios over time
    lr_inventory_by_sim = []
    lr_procurement_by_sim = []
    lr_other_by_sim = []

    # Individual compute production factors over time
    is_operational_by_sim = []
    wafer_starts_by_sim = []
    chips_per_wafer_by_sim = []
    architecture_efficiency_by_sim = []
    compute_per_wafer_2022_arch_by_sim = []
    transistor_density_by_sim = []
    process_node_by_sim = []

    # Use all simulations for visualization
    simulations_to_plot = model.simulation_results

    # Track survival rates, dark compute, operational dark compute, and datacenter capacity
    survival_rate_by_sim = []
    dark_compute_by_sim = []
    dark_compute_objects_by_sim = []  # Store Compute objects for energy breakdown
    operational_dark_compute_by_sim = []
    datacenter_capacity_by_sim = []
    lr_datacenters_by_sim = []

    for covert_projects, detectors in simulations_to_plot:
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        covert_fab = covert_projects["prc_covert_project"].covert_fab
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock
        covert_datacenters = covert_projects["prc_covert_project"].covert_datacenters

        years = sorted(us_beliefs.keys())
        if not all_years:
            all_years = years

        us_probs = [us_beliefs[year].p_covert_fab_exists for year in years]
        # Use covert fab's production over time if fab exists, otherwise 0
        if covert_fab is not None and hasattr(covert_fab, 'dark_compute_over_time'):
            h100e_counts = [covert_fab.dark_compute_over_time.get(year, 0.0) for year in years]
        else:
            h100e_counts = [0.0 for year in years]

        # Calculate survival rates, operational dark compute, and datacenter capacity for this simulation
        survival_rates = []
        dark_compute = []
        dark_compute_objects = []  # Store Compute objects for energy breakdown
        operational_dark_compute = []
        datacenter_capacity_gw = []
        lr_datacenters_over_time = []
        for year in years:
            # Get datacenter capacity in GW for this year
            # Years are relative to agreement year, so calculate relative year
            relative_year = year - agreement_year
            capacity_gw = covert_datacenters.get_GW_capacity(relative_year)
            datacenter_capacity_gw.append(capacity_gw)

            # Get compute stocks for survival rate calculation
            surviving_compute = dark_compute_stock.dark_compute(year)
            total_compute = dark_compute_stock.dark_compute_dead_and_alive(year)

            # Calculate survival rate (ratio of surviving to total, not limited by capacity)
            surviving = surviving_compute.total_h100e_tpp()
            total = total_compute.total_h100e_tpp()

            if total > 0:
                survival_rates.append(surviving / total)
            else:
                survival_rates.append(0.0)

            # Store dark compute (surviving, not limited by capacity)
            dark_compute.append(surviving)
            dark_compute_objects.append(surviving_compute)  # Store Compute object for energy breakdown

            # Get operational compute (limited by datacenter capacity)
            operational_compute = dark_compute_stock.operational_dark_compute(year, capacity_gw)
            operational = operational_compute.total_h100e_tpp()
            operational_dark_compute.append(operational)

            # Get likelihood ratio from datacenters for this year
            lr_datacenters = covert_datacenters.lr_from_concealed_datacenters(relative_year)
            lr_datacenters_over_time.append(lr_datacenters)

        us_probs_by_sim.append(us_probs)
        h100e_by_sim.append(h100e_counts)
        survival_rate_by_sim.append(survival_rates)
        dark_compute_by_sim.append(dark_compute)
        dark_compute_objects_by_sim.append(dark_compute_objects)
        operational_dark_compute_by_sim.append(operational_dark_compute)
        datacenter_capacity_by_sim.append(datacenter_capacity_gw)
        lr_datacenters_by_sim.append(lr_datacenters_over_time)

        # Extract likelihood ratio components
        if covert_fab is not None:
            # Use detection_updates which stores fab-specific detection LRs over time
            detection_lr_over_time = []
            lr_inventory_over_time = []
            lr_procurement_over_time = []
            lr_other_over_time = []
            is_operational = []
            wafer_starts = []
            architecture_efficiency = []
            compute_per_wafer_2022_arch = []

            for year in years:
                # Get the detection LR components from the fab's tracking dictionaries
                if hasattr(covert_fab, 'lr_inventory_over_time') and year in covert_fab.lr_inventory_over_time:
                    lr_inventory_over_time.append(covert_fab.lr_inventory_over_time[year])
                    lr_procurement_over_time.append(covert_fab.lr_procurement_over_time[year])
                    lr_other_over_time.append(covert_fab.lr_other_over_time[year])
                    detection_lr_over_time.append(covert_fab.detection_updates[year])
                else:
                    lr_inventory_over_time.append(1.0)
                    lr_procurement_over_time.append(1.0)
                    lr_other_over_time.append(1.0)
                    detection_lr_over_time.append(1.0)

                # Calculate compute factors
                is_operational.append(1.0 if covert_fab.is_operational(year) else 0.0)
                wafer_starts.append(covert_fab.wafer_starts_per_month)

                # Architecture efficiency and compute per wafer (2022 architectures)
                from fab_model import estimate_architecture_efficiency_relative_to_h100
                arch_efficiency = estimate_architecture_efficiency_relative_to_h100(year, agreement_year)
                architecture_efficiency.append(arch_efficiency)

                # Compute per wafer with 2022 architectures (without year-specific architecture improvements)
                compute_per_wafer_2022_arch.append(
                    covert_fab.h100_sized_chips_per_wafer *
                    covert_fab.transistor_density_relative_to_h100
                )

            lr_inventory_by_sim.append(lr_inventory_over_time)
            lr_procurement_by_sim.append(lr_procurement_over_time)
            lr_other_by_sim.append(lr_other_over_time)
            is_operational_by_sim.append(is_operational)
            wafer_starts_by_sim.append(wafer_starts)
            chips_per_wafer_by_sim.append([covert_fab.h100_sized_chips_per_wafer] * len(years))
            architecture_efficiency_by_sim.append(architecture_efficiency)
            compute_per_wafer_2022_arch_by_sim.append(compute_per_wafer_2022_arch)
            transistor_density_by_sim.append([covert_fab.transistor_density_relative_to_h100] * len(years))
            process_node_by_sim.append(covert_fab.process_node.value)  # Store the process node label (e.g., "130nm")

    # Calculate statistics
    years_array = np.array(all_years)
    us_probs_array = np.array(us_probs_by_sim)
    h100e_array = np.array(h100e_by_sim)
    survival_rate_array = np.array(survival_rate_by_sim)
    dark_compute_array = np.array(dark_compute_by_sim)
    operational_dark_compute_array = np.array(operational_dark_compute_by_sim)
    datacenter_capacity_array = np.array(datacenter_capacity_by_sim)

    us_probs_median = np.median(us_probs_array, axis=0)
    us_probs_p25 = np.percentile(us_probs_array, 25, axis=0)
    us_probs_p75 = np.percentile(us_probs_array, 75, axis=0)

    h100e_median = np.median(h100e_array, axis=0) / 1e3  # Convert to thousands
    h100e_p25 = np.percentile(h100e_array, 25, axis=0) / 1e3
    h100e_p75 = np.percentile(h100e_array, 75, axis=0) / 1e3

    survival_rate_median = np.median(survival_rate_array, axis=0)
    survival_rate_p25 = np.percentile(survival_rate_array, 25, axis=0)
    survival_rate_p75 = np.percentile(survival_rate_array, 75, axis=0)

    dark_compute_median = np.median(dark_compute_array, axis=0) / 1e3  # Convert to thousands
    dark_compute_p25 = np.percentile(dark_compute_array, 25, axis=0) / 1e3
    dark_compute_p75 = np.percentile(dark_compute_array, 75, axis=0) / 1e3

    operational_dark_compute_median = np.median(operational_dark_compute_array, axis=0) / 1e3  # Convert to thousands
    operational_dark_compute_p25 = np.percentile(operational_dark_compute_array, 25, axis=0) / 1e3
    operational_dark_compute_p75 = np.percentile(operational_dark_compute_array, 75, axis=0) / 1e3

    datacenter_capacity_median = np.median(datacenter_capacity_array, axis=0)
    datacenter_capacity_p25 = np.percentile(datacenter_capacity_array, 25, axis=0)
    datacenter_capacity_p75 = np.percentile(datacenter_capacity_array, 75, axis=0)

    lr_datacenters_array = np.array(lr_datacenters_by_sim)
    lr_datacenters_median = np.median(lr_datacenters_array, axis=0)
    lr_datacenters_p25 = np.percentile(lr_datacenters_array, 25, axis=0)
    lr_datacenters_p75 = np.percentile(lr_datacenters_array, 75, axis=0)

    # Calculate probability of detection (LR >= 5) for each year
    detection_threshold = 5.0
    datacenter_detection_prob = np.mean(lr_datacenters_array >= detection_threshold, axis=0)

    # Extract energy breakdown by source for ALL simulations, then compute median
    num_years = len(years_array)
    num_sims = len(simulations_to_plot)

    # Arrays to store energy by source for all simulations: [sim_idx, year_idx, source]
    initial_energy_all_sims = np.zeros((num_sims, num_years))
    fab_energy_all_sims = np.zeros((num_sims, num_years))
    initial_h100e_all_sims = np.zeros((num_sims, num_years))
    fab_h100e_all_sims = np.zeros((num_sims, num_years))

    for sim_idx, (covert_projects, _) in enumerate(simulations_to_plot):
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock

        for year_idx, year in enumerate(years_array):
            initial_energy, fab_energy, initial_h100e, fab_h100e = dark_compute_stock.dark_compute_energy_by_source(year)
            initial_energy_all_sims[sim_idx, year_idx] = initial_energy
            fab_energy_all_sims[sim_idx, year_idx] = fab_energy
            initial_h100e_all_sims[sim_idx, year_idx] = initial_h100e
            fab_h100e_all_sims[sim_idx, year_idx] = fab_h100e

    # Compute median across simulations for each year
    initial_energy_median = np.median(initial_energy_all_sims, axis=0)
    fab_energy_median = np.median(fab_energy_all_sims, axis=0)
    initial_h100e_median = np.median(initial_h100e_all_sims, axis=0)
    fab_h100e_median = np.median(fab_h100e_all_sims, axis=0)

    # Create energy_by_source_array using median values
    energy_by_source_array = np.zeros((num_years, 2))  # 2 sources: initial stock, fab
    energy_by_source_array[:, 0] = initial_energy_median
    energy_by_source_array[:, 1] = fab_energy_median

    # Calculate average efficiency using median values across all years
    initial_energy_total = np.sum(initial_energy_median)
    fab_energy_total = np.sum(fab_energy_median)
    initial_h100e_total = np.sum(initial_h100e_median)
    fab_h100e_total = np.sum(fab_h100e_median)

    # Calculate average efficiency: H100e TPP per GW
    initial_efficiency = (initial_h100e_total / initial_energy_total) if initial_energy_total > 0 else 0
    fab_efficiency = (fab_h100e_total / fab_energy_total) if fab_energy_total > 0 else 0

    # Calculate baseline H100 efficiency (H100e/GW for standard H100 with efficiency=1.0)
    # Energy per H100e = H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / 1e9 GW
    # H100e per GW = 1e9 / (H100_TPP_PER_CHIP * H100_WATTS_PER_TPP)
    h100_baseline_efficiency = 1e9 / (H100_TPP_PER_CHIP * H100_WATTS_PER_TPP)

    # Energy efficiency relative to H100 (higher is more efficient)
    initial_efficiency_relative = initial_efficiency / h100_baseline_efficiency if h100_baseline_efficiency > 0 else 0
    fab_efficiency_relative = fab_efficiency / h100_baseline_efficiency if h100_baseline_efficiency > 0 else 0

    source_labels = [
        f"Initial Dark Compute ({initial_efficiency_relative:.2f}x energy efficiency)",
        f"Covert Fab Compute ({fab_efficiency_relative:.2f}x energy efficiency)"
    ]

    # Extract compute before detection data for multiple thresholds
    # Calculate thresholds based on prior and likelihood ratios
    prior_odds = p_fab_exists / (1 - p_fab_exists)
    detection_thresholds = []
    for lr in LIKELIHOOD_RATIOS:
        posterior_odds = prior_odds * lr
        threshold = posterior_odds / (1 + posterior_odds)
        detection_thresholds.append(threshold)
    compute_ccdfs = {}
    op_time_ccdfs = {}

    # Store individual values for highest LR threshold (e.g., 5x update) (for dashboard)
    individual_h100e_before_detection = []
    individual_time_before_detection = []
    individual_process_nodes = []
    individual_energy_before_detection = []
    dashboard_threshold = detection_thresholds[-1]  # Use highest LR update threshold

    # Count total simulations with covert fab for correct CCDF calculation
    total_simulations_with_fab = sum(1 for cp, _ in model.simulation_results
                                     if cp["prc_covert_project"].covert_fab is not None)

    # Direct counters for the user's specific questions - calculate these FIRST, outside the detection loop
    # These should match what the plot shows
    fabs_never_finished_construction = 0
    fabs_finished_construction = 0
    fabs_finished_and_detected_before_finish = 0

    for covert_projects, detectors in model.simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab
        if covert_fab is None:
            continue

        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        years = sorted(us_beliefs.keys())
        final_year = all_years[-1]  # Use all_years to match the plot

        construction_start = covert_fab.construction_start_year
        construction_duration = covert_fab.construction_duration
        operational_start = construction_start + construction_duration

        # Check if fab is operational at the last simulation timestep
        # This matches the is_operational() logic used in the plot
        if final_year >= operational_start:
            fabs_finished_construction += 1

            # Check if detected before finishing construction using fab-only detection
            detection_year = None
            initial_p_fab = us_beliefs[min(years)].p_covert_fab_exists
            prior_odds_fab = initial_p_fab / (1 - initial_p_fab)

            for year in years:
                # Calculate fab-only detection probability
                if hasattr(covert_fab, 'detection_updates') and year in covert_fab.detection_updates:
                    fab_lr = covert_fab.detection_updates[year]
                    posterior_odds_fab = prior_odds_fab * fab_lr
                    p_fab_detected = posterior_odds_fab / (1 + posterior_odds_fab)

                    if p_fab_detected >= 0.5:
                        detection_year = year
                        break

            if detection_year is not None and detection_year < operational_start:
                fabs_finished_and_detected_before_finish += 1
        else:
            fabs_never_finished_construction += 1

    # Track fabs detected during construction (for dashboard threshold only)
    detected_during_construction_count = 0
    detected_after_operational_count = 0
    never_detected_count = 0
    never_operational_count = 0

    for threshold in detection_thresholds:
        compute_at_detection = []
        operational_time_at_detection = []

        for covert_projects, detectors in model.simulation_results:
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            covert_fab = covert_projects["prc_covert_project"].covert_fab

            if covert_fab is None:
                continue

            # Get fab production over time
            h100e_over_time = covert_fab.dark_compute_over_time if hasattr(covert_fab, 'dark_compute_over_time') else {}

            years = sorted(us_beliefs.keys())
            detection_year = None

            # Calculate fab-only detection probability using only fab-specific detection updates
            # Start with the initial prior for covert fab
            initial_p_fab = us_beliefs[min(years)].p_covert_fab_exists
            prior_odds_fab = initial_p_fab / (1 - initial_p_fab)

            for year in years:
                # Get cumulative likelihood ratio from fab detection updates only
                if hasattr(covert_fab, 'detection_updates') and year in covert_fab.detection_updates:
                    fab_lr = covert_fab.detection_updates[year]
                    # Calculate posterior probability using only fab-specific evidence
                    posterior_odds_fab = prior_odds_fab * fab_lr
                    p_fab_detected = posterior_odds_fab / (1 + posterior_odds_fab)

                    if p_fab_detected >= threshold:
                        detection_year = year
                        break

            if detection_year is not None:
                # Find the most recent year <= detection_year in h100e_over_time
                available_years = [y for y in h100e_over_time.keys() if y <= detection_year]
                if available_years:
                    most_recent_year = max(available_years)
                    h100e_at_detection = h100e_over_time[most_recent_year]
                else:
                    h100e_at_detection = 0.0
                compute_at_detection.append(h100e_at_detection)

                construction_start = covert_fab.construction_start_year
                construction_duration = covert_fab.construction_duration
                operational_start = construction_start + construction_duration

                if detection_year >= operational_start:
                    operational_time = detection_year - operational_start
                else:
                    operational_time = 0.0

                operational_time_at_detection.append(operational_time)

                # Track detection timing (for dashboard threshold only)
                if threshold == dashboard_threshold:
                    if detection_year <= operational_start:
                        detected_during_construction_count += 1
                        # Check if construction hasn't even finished yet
                        if detection_year < operational_start:
                            never_operational_count += 1
                    else:
                        detected_after_operational_count += 1

                    # Debug: Check fabs detected after operational but with 0 compute
                    if detection_year >= operational_start and h100e_at_detection <= 0.0:
                        available_years = sorted(h100e_over_time.keys())
                        print(f"DEBUG: Fab detected after operational with 0 compute - detection_year={detection_year}, operational_start={operational_start}, h100e={h100e_at_detection}, available_years={available_years}", flush=True)

                # Store individual values for dashboard threshold (for dashboard)
                if threshold == dashboard_threshold:
                    individual_h100e_before_detection.append(h100e_at_detection)
                    individual_time_before_detection.append(operational_time)
                    individual_process_nodes.append(covert_fab.process_node.value)

                    # Calculate energy (GW) from H100e TPP
                    # Formula: energy_gw = h100e_tpp * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / energy_efficiency / 1e9
                    from stock_model import InitialPRCComputeStockParameters, H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
                    energy_gw = (h100e_at_detection * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                                InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100 / 1e9)
                    individual_energy_before_detection.append(energy_gw)

                    # Debug: Check fabs that finished construction but have 0 compute
                    if detection_year >= operational_start and h100e_at_detection <= 0.0:
                        print(f"DEBUG: Fab finished construction but has 0 compute - detection_year={detection_year}, operational_start={operational_start}, time_since_operational={detection_year - operational_start:.4f}, available_years_count={len(h100e_over_time)}", flush=True)
            else:
                # Detection never happened - include final values in CCDF
                # Use all_years (from first simulation) to match what the plot uses
                final_year = all_years[-1]
                # Find the most recent year <= final_year in h100e_over_time
                available_years = [y for y in h100e_over_time.keys() if y <= final_year]
                if available_years:
                    most_recent_year = max(available_years)
                    h100e_at_end = h100e_over_time[most_recent_year]
                else:
                    h100e_at_end = 0.0
                compute_at_detection.append(h100e_at_end)

                construction_start = covert_fab.construction_start_year
                construction_duration = covert_fab.construction_duration
                operational_start = construction_start + construction_duration

                if final_year >= operational_start:
                    operational_time = final_year - operational_start
                else:
                    operational_time = 0.0

                operational_time_at_detection.append(operational_time)

                # Track never detected (for dashboard threshold only)
                if threshold == dashboard_threshold:
                    never_detected_count += 1

                    # Check if never became operational
                    if final_year < operational_start:
                        never_operational_count += 1

                    # Debug: Check never-detected fabs with 0 compute
                    if h100e_at_end <= 0.0:
                        available_years = sorted(h100e_over_time.keys())
                        print(f"DEBUG: Never-detected fab with 0 compute - final_year={final_year}, operational_start={operational_start}, construction_start={construction_start}, duration={construction_duration}, h100e={h100e_at_end}, available_years={available_years}", flush=True)

                # Also store for dashboard (dashboard threshold only)
                if threshold == dashboard_threshold:
                    individual_h100e_before_detection.append(h100e_at_end)
                    individual_time_before_detection.append(operational_time)
                    individual_process_nodes.append(covert_fab.process_node.value)

                    # Calculate energy (GW) from H100e TPP
                    from stock_model import InitialPRCComputeStockParameters, H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
                    energy_gw = (h100e_at_end * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                                InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100 / 1e9)
                    individual_energy_before_detection.append(energy_gw)

        # Calculate CCDFs for this threshold
        # Use total_simulations_with_fab as denominator to account for runs where detection never happened
        if compute_at_detection:
            compute_sorted = np.sort(compute_at_detection)

            # Handle ties correctly: CCDF should be fraction of values STRICTLY GREATER than x
            # For each unique x value, find how many values are > x
            unique_x = []
            ccdf_y = []
            seen_values = set()

            for i, x in enumerate(compute_sorted):
                if x not in seen_values:
                    seen_values.add(x)
                    # Count how many values are strictly greater than x
                    num_greater = total_simulations_with_fab - (i + 1)
                    # Find last occurrence of this x value
                    last_idx = i
                    while last_idx + 1 < len(compute_sorted) and compute_sorted[last_idx + 1] == x:
                        last_idx += 1
                    # CCDF at x = (number of values > x) / total
                    num_greater = total_simulations_with_fab - (last_idx + 1)
                    ccdf = num_greater / total_simulations_with_fab
                    unique_x.append(float(x))
                    ccdf_y.append(float(ccdf))

            compute_ccdfs[threshold] = [{"x": x, "y": y} for x, y in zip(unique_x, ccdf_y)]

        if operational_time_at_detection:
            op_time_sorted = np.sort(operational_time_at_detection)

            # Handle ties correctly for operational time CCDF too
            unique_x_time = []
            ccdf_y_time = []
            seen_values_time = set()

            for i, x in enumerate(op_time_sorted):
                if x not in seen_values_time:
                    seen_values_time.add(x)
                    # Find last occurrence of this x value
                    last_idx = i
                    while last_idx + 1 < len(op_time_sorted) and op_time_sorted[last_idx + 1] == x:
                        last_idx += 1
                    # CCDF at x = (number of values > x) / total
                    num_greater = total_simulations_with_fab - (last_idx + 1)
                    ccdf = num_greater / total_simulations_with_fab
                    unique_x_time.append(float(x))
                    ccdf_y_time.append(float(ccdf))

            op_time_ccdfs[threshold] = [{"x": x, "y": y} for x, y in zip(unique_x_time, ccdf_y_time)]

    # Keep backward compatibility with single threshold
    compute_ccdf = compute_ccdfs.get(dashboard_threshold, [])
    op_time_ccdf = op_time_ccdfs.get(dashboard_threshold, [])

    # Track dark compute project detection (for dashboard)
    # We track operational dark compute at detection for each simulation
    individual_project_h100e_before_detection = []
    individual_project_energy_before_detection = []
    individual_project_time_before_detection = []
    individual_project_h100_years_before_detection = []

    for sim_idx, (covert_projects, detectors) in enumerate(model.simulation_results):
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock
        covert_datacenters = covert_projects["prc_covert_project"].covert_datacenters

        years = sorted(us_beliefs.keys())
        detection_year = None

        # Get initial prior for covert project (not just fab)
        initial_p_project = us_beliefs[min(years)].p_project_exists
        prior_odds_project = initial_p_project / (1 - initial_p_project) if initial_p_project < 1.0 else 1e10

        # Find detection year based on project probability and odds ratio
        for year in years:
            p_project = us_beliefs[year].p_project_exists
            # Detection occurs when odds ratio >= threshold (dashboard_threshold)
            current_odds = p_project / (1 - p_project) if p_project < 1.0 else 1e10
            odds_ratio = current_odds / prior_odds_project

            if odds_ratio >= dashboard_threshold:
                detection_year = year
                break

        # If detected, get operational dark compute at detection
        if detection_year is not None:
            # Get datacenter capacity at detection year
            capacity_gw = covert_datacenters.get_GW_capacity(detection_year - agreement_year)

            # Get operational dark compute (limited by capacity)
            operational_compute = dark_compute_stock.operational_dark_compute(
                detection_year - agreement_year,
                capacity_gw
            )
            operational_h100e = operational_compute.total_h100e_tpp()

            # Calculate energy (GW) from H100e TPP
            from stock_model import InitialPRCComputeStockParameters, H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
            energy_gw = (operational_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                        InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100 / 1e9)

            # Calculate time operational (detection year - agreement start)
            time_operational = detection_year - agreement_year

            # Calculate H100-years: integrate operational H100e over time until detection
            # For each year from agreement start to detection, calculate operational H100e
            h100_years = 0.0
            years_to_integrate = [y for y in years if agreement_year <= y <= detection_year]
            for i in range(len(years_to_integrate) - 1):
                year = years_to_integrate[i]
                next_year = years_to_integrate[i + 1]
                time_increment = next_year - year

                # Get operational H100e at this time
                year_since_agreement = year - agreement_year
                capacity_at_year = covert_datacenters.get_GW_capacity(year_since_agreement)
                operational_at_year = dark_compute_stock.operational_dark_compute(
                    year_since_agreement,
                    capacity_at_year
                )
                h100e_at_year = operational_at_year.total_h100e_tpp()

                # Add contribution: H100e * time_increment (in years)
                h100_years += h100e_at_year * time_increment

            individual_project_h100e_before_detection.append(operational_h100e)
            individual_project_energy_before_detection.append(energy_gw)
            individual_project_time_before_detection.append(time_operational)
            individual_project_h100_years_before_detection.append(h100_years)
        else:
            # Never detected - use final year values
            final_year = max(years)
            capacity_gw = covert_datacenters.get_GW_capacity(final_year - agreement_year)
            operational_compute = dark_compute_stock.operational_dark_compute(
                final_year - agreement_year,
                capacity_gw
            )
            operational_h100e = operational_compute.total_h100e_tpp()

            from stock_model import InitialPRCComputeStockParameters, H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
            energy_gw = (operational_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                        InitialPRCComputeStockParameters.energy_efficiency_relative_to_h100 / 1e9)

            time_operational = final_year - agreement_year

            # Calculate H100-years over entire simulation period
            h100_years = 0.0
            years_to_integrate = [y for y in years if agreement_year <= y]
            for i in range(len(years_to_integrate) - 1):
                year = years_to_integrate[i]
                next_year = years_to_integrate[i + 1]
                time_increment = next_year - year

                # Get operational H100e at this time
                year_since_agreement = year - agreement_year
                capacity_at_year = covert_datacenters.get_GW_capacity(year_since_agreement)
                operational_at_year = dark_compute_stock.operational_dark_compute(
                    year_since_agreement,
                    capacity_at_year
                )
                h100e_at_year = operational_at_year.total_h100e_tpp()

                # Add contribution: H100e * time_increment (in years)
                h100_years += h100e_at_year * time_increment

            individual_project_h100e_before_detection.append(operational_h100e)
            individual_project_energy_before_detection.append(energy_gw)
            individual_project_time_before_detection.append(time_operational)
            individual_project_h100_years_before_detection.append(h100_years)

    # Calculate CCDFs for H100-years at different detection thresholds
    # Similar to compute_ccdfs but for dark compute project
    project_h100_years_ccdfs = {}
    total_simulations = len(model.simulation_results)

    # For each threshold (likelihood ratio), collect H100-years at detection
    for lr in LIKELIHOOD_RATIOS:
        posterior_odds = prior_odds_project * lr
        threshold = posterior_odds / (1 + posterior_odds)

        h100_years_at_threshold = []

        # Go through each simulation and find detection year for this threshold
        for sim_idx, (covert_projects, detectors) in enumerate(model.simulation_results):
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock
            covert_datacenters = covert_projects["prc_covert_project"].covert_datacenters

            years = sorted(us_beliefs.keys())
            detection_year = None

            # Get initial prior for covert project
            initial_p_project = us_beliefs[min(years)].p_project_exists
            prior_odds = initial_p_project / (1 - initial_p_project) if initial_p_project < 1.0 else 1e10

            # Find detection year based on project probability and odds ratio
            for year in years:
                p_project = us_beliefs[year].p_project_exists
                current_odds = p_project / (1 - p_project) if p_project < 1.0 else 1e10
                odds_ratio = current_odds / prior_odds

                if odds_ratio >= lr:
                    detection_year = year
                    break

            # Calculate H100-years until detection (or end of simulation)
            end_year = detection_year if detection_year is not None else max(years)
            h100_years = 0.0
            years_to_integrate = [y for y in years if agreement_year <= y <= end_year]

            for i in range(len(years_to_integrate) - 1):
                year = years_to_integrate[i]
                next_year = years_to_integrate[i + 1]
                time_increment = next_year - year

                # Get operational H100e at this time
                year_since_agreement = year - agreement_year
                capacity_at_year = covert_datacenters.get_GW_capacity(year_since_agreement)
                operational_at_year = dark_compute_stock.operational_dark_compute(
                    year_since_agreement,
                    capacity_at_year
                )
                h100e_at_year = operational_at_year.total_h100e_tpp()

                # Add contribution: H100e * time_increment (in years)
                h100_years += h100e_at_year * time_increment

            h100_years_at_threshold.append(h100_years)

        # Calculate CCDF for this threshold
        if h100_years_at_threshold:
            h100_years_sorted = np.sort(h100_years_at_threshold)

            # Handle ties correctly
            unique_x = []
            ccdf_y = []
            seen_values = set()

            for i, x in enumerate(h100_years_sorted):
                if x not in seen_values:
                    seen_values.add(x)
                    # Find last occurrence of this x value
                    last_idx = i
                    while last_idx + 1 < len(h100_years_sorted) and h100_years_sorted[last_idx + 1] == x:
                        last_idx += 1
                    # CCDF at x = (number of values > x) / total
                    num_greater = total_simulations - (last_idx + 1)
                    ccdf = num_greater / total_simulations
                    unique_x.append(float(x))
                    ccdf_y.append(float(ccdf))

            project_h100_years_ccdfs[threshold] = [{"x": x, "y": y} for x, y in zip(unique_x, ccdf_y)]

    # Print USER'S DIRECT QUESTIONS FIRST
    print(f"\n=== DIRECT ANSWERS TO USER'S QUESTIONS ===", flush=True)
    if total_simulations_with_fab > 0:
        print(f"1. Fabs that NEVER finished construction before simulation end: {fabs_never_finished_construction} ({100*fabs_never_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)
        print(f"2. Fabs that DID finish construction before simulation end: {fabs_finished_construction} ({100*fabs_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)
        print(f"3. Of those that finished, detected BEFORE finishing construction: {fabs_finished_and_detected_before_finish} ({100*fabs_finished_and_detected_before_finish/fabs_finished_construction:.1f}% of finished fabs)" if fabs_finished_construction > 0 else "3. Of those that finished, detected BEFORE finishing construction: 0", flush=True)
    else:
        print(f"No fab simulations (build_covert_fab is disabled)", flush=True)
    print(f"==========================================\n", flush=True)

    # Print detection timing analysis
    print(f"\nDETECTION TIMING ANALYSIS (P(fab) >= 50%):", flush=True)
    print(f"  Total simulations with fab: {total_simulations_with_fab}", flush=True)
    if total_simulations_with_fab > 0:
        print(f"  Detected during construction: {detected_during_construction_count} ({100*detected_during_construction_count/total_simulations_with_fab:.1f}%)", flush=True)
        print(f"  Detected after finished construction: {detected_after_operational_count} ({100*detected_after_operational_count/total_simulations_with_fab:.1f}%)", flush=True)
        print(f"  Never detected: {never_detected_count} ({100*never_detected_count/total_simulations_with_fab:.1f}%)", flush=True)
        print(f"  Fabs that never finished construction (should equal fabs never finished above): {fabs_never_finished_construction} ({100*fabs_never_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)
    else:
        print(f"  No fab simulations to analyze", flush=True)

    # Debug: Compare CCDF data with dashboard data
    if individual_h100e_before_detection:
        sorted_individual = sorted(individual_h100e_before_detection)
        idx80_dash = int(len(sorted_individual) * 0.8)
        print(f"\nDEBUG H100e COMPARISON:", flush=True)
        print(f"  Total sims with fab: {total_simulations_with_fab}", flush=True)
        print(f"  Dashboard individual values (includes non-detected): {len(individual_h100e_before_detection)}", flush=True)
        print(f"  Dashboard 80th percentile: {sorted_individual[idx80_dash]:.0f}", flush=True)

        # Analyze H100e values for compute > 0
        if individual_h100e_before_detection:
            num_zero_compute = sum(1 for h in individual_h100e_before_detection if h <= 0.0)
            num_positive_compute = sum(1 for h in individual_h100e_before_detection if h > 0.0)
            total = len(individual_h100e_before_detection)
            p_compute_gt_0 = num_positive_compute / total if total > 0 else 0

            print(f"\nCOMPUTE BEFORE DETECTION ANALYSIS:", flush=True)
            print(f"  Fabs with 0 compute at detection: {num_zero_compute} ({100*num_zero_compute/total:.1f}%)", flush=True)
            print(f"  Fabs with >0 compute at detection: {num_positive_compute} ({100*num_positive_compute/total:.1f}%)", flush=True)
            print(f"  P(covert compute > 0) = {p_compute_gt_0:.4f} ({100*p_compute_gt_0:.1f}%)", flush=True)

            # Also check CCDF first point
            if compute_ccdf and len(compute_ccdf) > 0:
                print(f"  CCDF smallest x value: {compute_ccdf[0]['x']:.2f} H100e", flush=True)
                print(f"  CCDF at smallest x: P(compute > {compute_ccdf[0]['x']:.2f}) = {compute_ccdf[0]['y']:.4f}", flush=True)

    # Calculate statistics for fab detection LR
    lr_components = {}
    if lr_inventory_by_sim:
        lr_fab_detection_array = np.array(lr_inventory_by_sim)

        # Analyze final timestep fab detection LR distribution
        lr_fab_detection_final = [sim[-1] for sim in lr_inventory_by_sim]
        num_lr_equals_1 = sum(1 for lr in lr_fab_detection_final if abs(lr - 1.0) < 0.01)
        num_lr_gte_10 = sum(1 for lr in lr_fab_detection_final if lr >= 10.0)
        num_lr_gt_2_5 = sum(1 for lr in lr_fab_detection_final if lr > 2.5)
        total_sims = len(lr_fab_detection_final)
        print(f"FAB DETECTION LR ANALYSIS (final timestep):", flush=True)
        print(f"  Total simulations: {total_sims}", flush=True)
        print(f"  LR ≈ 1.0: {num_lr_equals_1} ({100*num_lr_equals_1/total_sims:.1f}%)", flush=True)
        print(f"  LR > 2.5: {num_lr_gt_2_5} ({100*num_lr_gt_2_5/total_sims:.1f}%)", flush=True)
        print(f"  LR ≥ 10.0: {num_lr_gte_10} ({100*num_lr_gte_10/total_sims:.1f}%)", flush=True)
        print(f"  Other values: {total_sims - num_lr_equals_1 - num_lr_gte_10} ({100*(total_sims - num_lr_equals_1 - num_lr_gte_10)/total_sims:.1f}%)", flush=True)

        # Calculate medians and prepare data for all three components
        lr_inventory_array = np.array(lr_inventory_by_sim)
        lr_procurement_array = np.array(lr_procurement_by_sim)
        lr_other_array = np.array(lr_other_by_sim)

        lr_components = {
            "inventory_median": np.median(lr_inventory_array, axis=0).tolist(),
            "inventory_individual": [list(sim) for sim in lr_inventory_by_sim],
            "procurement_median": np.median(lr_procurement_array, axis=0).tolist(),
            "procurement_individual": [list(sim) for sim in lr_procurement_by_sim],
            "other_median": np.median(lr_other_array, axis=0).tolist(),
            "other_individual": [list(sim) for sim in lr_other_by_sim]
        }

    # Calculate statistics for compute factors
    compute_factors = {}
    if is_operational_by_sim:
        is_operational_array = np.array(is_operational_by_sim)
        wafer_starts_array = np.array(wafer_starts_by_sim)
        chips_per_wafer_array = np.array(chips_per_wafer_by_sim)
        architecture_efficiency_array = np.array(architecture_efficiency_by_sim)
        compute_per_wafer_2022_arch_array = np.array(compute_per_wafer_2022_arch_by_sim)
        transistor_density_array = np.array(transistor_density_by_sim)

        # Debug: Check construction completion
        print(f"\n=== DEBUG CONSTRUCTION COMPLETION ===", flush=True)
        print(f"Total simulations in model.simulation_results: {len(model.simulation_results)}", flush=True)
        print(f"Number of entries in is_operational_by_sim: {len(is_operational_by_sim)}", flush=True)

        # Count simulations with and without fabs
        num_with_fab = sum(1 for cp, _ in model.simulation_results if cp["prc_covert_project"].covert_fab is not None)
        num_without_fab = len(model.simulation_results) - num_with_fab
        print(f"Simulations WITH fab: {num_with_fab}", flush=True)
        print(f"Simulations WITHOUT fab: {num_without_fab}", flush=True)

        print(f"\nYears array length: {len(all_years)}", flush=True)
        print(f"First years: {all_years[:5]}", flush=True)
        print(f"Last years: {all_years[-5:]}", flush=True)
        print(f"Last year: {all_years[-1]}", flush=True)
        print(f"\nis_operational_array shape: {is_operational_array.shape}", flush=True)
        print(f"is_operational at last timestep - mean across sims WITH FABS: {np.mean(is_operational_array[:, -1]):.4f}", flush=True)
        print(f"is_operational at last timestep - sum: {np.sum(is_operational_array[:, -1])}", flush=True)
        print(f"is_operational at last timestep - count of 1s: {np.sum(is_operational_array[:, -1] == 1.0)}", flush=True)
        print(f"is_operational at last timestep - count of 0s: {np.sum(is_operational_array[:, -1] == 0.0)}", flush=True)

        # Check construction end times
        construction_end_times = []
        for covert_projects, _ in model.simulation_results:
            fab = covert_projects["prc_covert_project"].covert_fab
            if fab:
                construction_end_times.append(fab.construction_start_year + fab.construction_duration)
        print(f"\nConstruction end times - min: {min(construction_end_times):.4f}, max: {max(construction_end_times):.4f}, mean: {np.mean(construction_end_times):.4f}", flush=True)
        print(f"Number of fabs finishing AFTER last simulation year ({all_years[-1]}): {sum(1 for t in construction_end_times if t > all_years[-1])}", flush=True)

        # Check which fabs are not operational at the last timestep
        print(f"\nChecking fabs NOT operational at last timestep:", flush=True)
        non_operational_count = 0
        for i, (covert_projects, _) in enumerate(model.simulation_results):
            fab = covert_projects["prc_covert_project"].covert_fab
            if fab is not None:
                construction_end = fab.construction_start_year + fab.construction_duration
                is_op = fab.is_operational(all_years[-1])
                if not is_op:
                    non_operational_count += 1
                    if non_operational_count <= 5:  # Only print first 5
                        print(f"  Sim {i}: construction_end={construction_end:.6f}, last_year={all_years[-1]:.6f}, diff={construction_end - all_years[-1]:.6f}, is_operational={is_op}", flush=True)
        print(f"Total non-operational at last timestep: {non_operational_count}", flush=True)
        print(f"=====================================\n", flush=True)

        compute_factors = {
            "is_operational_median": np.mean(is_operational_array, axis=0).tolist(),
            "is_operational_individual": [list(sim) for sim in is_operational_by_sim],
            "wafer_starts_median": np.median(wafer_starts_array, axis=0).tolist(),
            "wafer_starts_individual": [list(sim) for sim in wafer_starts_by_sim],
            "chips_per_wafer_median": np.median(chips_per_wafer_array, axis=0).tolist(),
            "chips_per_wafer_individual": [list(sim) for sim in chips_per_wafer_by_sim],
            "architecture_efficiency_median": np.median(architecture_efficiency_array, axis=0).tolist(),
            "architecture_efficiency_individual": [list(sim) for sim in architecture_efficiency_by_sim],
            "compute_per_wafer_2022_arch_median": np.median(compute_per_wafer_2022_arch_array, axis=0).tolist(),
            "compute_per_wafer_2022_arch_individual": [list(sim) for sim in compute_per_wafer_2022_arch_by_sim],
            "transistor_density_median": np.median(transistor_density_array, axis=0).tolist(),
            "transistor_density_individual": [list(sim) for sim in transistor_density_by_sim],
            "process_node_by_sim": process_node_by_sim
        }

    # Calculate architecture efficiency at agreement year
    from fab_model import estimate_architecture_efficiency_relative_to_h100
    architecture_efficiency_at_agreement = estimate_architecture_efficiency_relative_to_h100(agreement_year, agreement_year)

    return {
        "time_series": {
            "years": years_array.tolist(),
            "us_prob_median": us_probs_median.tolist(),
            "us_prob_p25": us_probs_p25.tolist(),
            "us_prob_p75": us_probs_p75.tolist(),
            "h100e_median": h100e_median.tolist(),
            "h100e_p25": h100e_p25.tolist(),
            "h100e_p75": h100e_p75.tolist(),
            "survival_rate_median": survival_rate_median.tolist(),
            "survival_rate_p25": survival_rate_p25.tolist(),
            "survival_rate_p75": survival_rate_p75.tolist(),
            "dark_compute_median": dark_compute_median.tolist(),
            "dark_compute_p25": dark_compute_p25.tolist(),
            "dark_compute_p75": dark_compute_p75.tolist(),
            "operational_dark_compute_median": operational_dark_compute_median.tolist(),
            "operational_dark_compute_p25": operational_dark_compute_p25.tolist(),
            "operational_dark_compute_p75": operational_dark_compute_p75.tolist(),
            "datacenter_capacity_median": datacenter_capacity_median.tolist(),
            "datacenter_capacity_p25": datacenter_capacity_p25.tolist(),
            "datacenter_capacity_p75": datacenter_capacity_p75.tolist(),
            "lr_datacenters_median": lr_datacenters_median.tolist(),
            "lr_datacenters_p25": lr_datacenters_p25.tolist(),
            "lr_datacenters_p75": lr_datacenters_p75.tolist(),
            "datacenter_detection_prob": datacenter_detection_prob.tolist(),
            "energy_by_source": energy_by_source_array.tolist(),
            "source_labels": source_labels,
            "individual_us_probs": [list(sim) for sim in us_probs_by_sim],
            "individual_h100e": [list(np.array(sim) / 1e3) for sim in h100e_by_sim]
        },
        "compute_ccdf": compute_ccdf,
        "compute_ccdfs": compute_ccdfs,
        "op_time_ccdf": op_time_ccdf,
        "op_time_ccdfs": op_time_ccdfs,
        "likelihood_ratios": LIKELIHOOD_RATIOS,
        "lr_components": lr_components,
        "compute_factors": compute_factors,
        "architecture_efficiency_at_agreement": architecture_efficiency_at_agreement,
        "num_simulations": len(model.simulation_results),
        "individual_h100e_before_detection": individual_h100e_before_detection,
        "individual_time_before_detection": individual_time_before_detection,
        "individual_process_node": individual_process_nodes,
        "individual_energy_before_detection": individual_energy_before_detection,
        "individual_project_h100e_before_detection": individual_project_h100e_before_detection,
        "individual_project_energy_before_detection": individual_project_energy_before_detection,
        "individual_project_time_before_detection": individual_project_time_before_detection,
        "individual_project_h100_years_before_detection": individual_project_h100_years_before_detection,
        "project_h100_years_ccdfs": project_h100_years_ccdfs,
        "p_project_exists": p_project_exists
    }

@app.route('/download/nuclear_case_studies')
def download_nuclear_case_studies():
    """Download the nuclear case studies CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'nuclear_case_studies.csv')
    return send_file(file_path, as_attachment=True, download_name='nuclear_case_studies.csv')

@app.route('/download/prc_indigenous_sme_capabilities')
def download_prc_indigenous_sme_capabilities():
    """Download the PRC indigenous SME capabilities CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'prc_indigenous_sme_capabilities.csv')
    return send_file(file_path, as_attachment=True, download_name='prc_indigenous_sme_capabilities.csv')

@app.route('/download/us_intelligence_estimates')
def download_us_intelligence_estimates():
    """Download the US intelligence estimates CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'us_intelligence_estimates.csv')
    return send_file(file_path, as_attachment=True, download_name='us_intelligence_estimates.csv')

@app.route('/download/compute_production_vs_operating_labor')
def download_compute_production_vs_operating_labor():
    """Download the compute production vs operating labor CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'compute_production_vs_operating_labor.csv')
    return send_file(file_path, as_attachment=True, download_name='compute_production_vs_operating_labor.csv')

@app.route('/download/asml_sales_history')
def download_asml_sales_history():
    """Download the ASML sales history CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'asml_sales_history.csv')
    return send_file(file_path, as_attachment=True, download_name='asml_sales_history.csv')

@app.route('/download/transistor_density_vs_node')
def download_transistor_density_vs_node():
    """Download the transistor density vs node CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'transistor_density_vs_node.csv')
    return send_file(file_path, as_attachment=True, download_name='transistor_density_vs_node.csv')

@app.route('/download/construction_time_vs_fab_capacity')
def download_construction_time_vs_fab_capacity():
    """Download the construction time vs fab capacity CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'construction_time_vs_fab_capacity.csv')
    return send_file(file_path, as_attachment=True, download_name='construction_time_vs_fab_capacity.csv')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
