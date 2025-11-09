from flask import Flask, render_template, jsonify, request, send_file
from model import Model, CovertProjectStrategy, default_prc_covert_project_strategy
from fab_model import ProcessNode, FabModelParameters
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    # Pass default strategy values to template
    defaults = {
        'operating_labor': default_prc_covert_project_strategy.covert_fab_operating_labor,
        'construction_labor': default_prc_covert_project_strategy.covert_fab_construction_labor,
        'scanner_proportion': default_prc_covert_project_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted,
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

    # Fab model parameters - update all parameters from data
    if 'median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock' in data:
        FabModelParameters.median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock = float(data['median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock'])
    if 'mean_detection_time_for_100_workers' in data:
        FabModelParameters.mean_detection_time_for_100_workers = float(data['mean_detection_time_for_100_workers'])
    if 'mean_detection_time_for_1000_workers' in data:
        FabModelParameters.mean_detection_time_for_1000_workers = float(data['mean_detection_time_for_1000_workers'])
    if 'variance_of_detection_time_given_num_workers' in data:
        FabModelParameters.variance_of_detection_time_given_num_workers = float(data['variance_of_detection_time_given_num_workers'])
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

        # Extract data for plots
        print(f"DEBUG: Extracting plot data...", flush=True)
        results = extract_plot_data(model)

        print(f"DEBUG: Returning results...", flush=True)
        return jsonify(results)
    except Exception as e:
        print(f"ERROR running simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def extract_plot_data(model):
    """Extract plot data from model simulation results"""
    if not model.simulation_results:
        return {"error": "No simulation results"}

    agreement_year = model.year_us_prc_agreement_goes_into_force

    # Extract time series data (like simulation_runs.png)
    all_years = []
    us_probs_by_sim = []
    h100e_by_sim = []

    # Individual likelihood ratio components over time
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

    for covert_projects, detectors in simulations_to_plot:
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        h100e_over_time = covert_projects["prc_covert_project"].h100e_over_time
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        years = sorted(us_beliefs.keys())
        if not all_years:
            all_years = years

        us_probs = [us_beliefs[year].p_covert_fab_exists for year in years]
        h100e_counts = [h100e_over_time.get(year, 0.0) for year in years]

        us_probs_by_sim.append(us_probs)
        h100e_by_sim.append(h100e_counts)

        # Extract likelihood ratio components
        if covert_fab is not None:
            lr_inventory = []
            lr_procurement = []
            lr_other = []
            is_operational = []
            wafer_starts = []
            architecture_efficiency = []
            compute_per_wafer_2022_arch = []

            for year in years:
                # Trigger calculation of LRs by calling detection_likelihood_ratio
                _ = covert_fab.detection_likelihood_ratio(year)
                lr_inventory.append(covert_fab.lr_inventory)
                lr_procurement.append(covert_fab.lr_procurement)
                lr_other.append(covert_fab.lr_other)

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

            lr_inventory_by_sim.append(lr_inventory)
            lr_procurement_by_sim.append(lr_procurement)
            lr_other_by_sim.append(lr_other)
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

    us_probs_median = np.median(us_probs_array, axis=0)
    us_probs_p25 = np.percentile(us_probs_array, 25, axis=0)
    us_probs_p75 = np.percentile(us_probs_array, 75, axis=0)

    h100e_median = np.median(h100e_array, axis=0) / 1e3  # Convert to thousands
    h100e_p25 = np.percentile(h100e_array, 25, axis=0) / 1e3
    h100e_p75 = np.percentile(h100e_array, 75, axis=0) / 1e3

    # Extract compute before detection data for multiple thresholds
    detection_thresholds = [0.5, 0.25, 0.125]
    compute_ccdfs = {}
    op_time_ccdfs = {}

    # Store individual values for 0.5 threshold (for dashboard)
    individual_h100e_before_detection = []
    individual_time_before_detection = []
    individual_process_nodes = []

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

            # Check if detected before finishing construction
            detection_year = None
            for year in years:
                if us_beliefs[year].p_covert_fab_exists >= 0.5:
                    detection_year = year
                    break

            if detection_year is not None and detection_year < operational_start:
                fabs_finished_and_detected_before_finish += 1
        else:
            fabs_never_finished_construction += 1

    # Track fabs detected during construction (for 0.5 threshold only)
    detected_during_construction_count = 0
    detected_after_operational_count = 0
    never_detected_count = 0
    never_operational_count = 0

    for threshold in detection_thresholds:
        compute_at_detection = []
        operational_time_at_detection = []

        for covert_projects, detectors in model.simulation_results:
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            h100e_over_time = covert_projects["prc_covert_project"].h100e_over_time
            covert_fab = covert_projects["prc_covert_project"].covert_fab

            if covert_fab is None:
                continue

            years = sorted(us_beliefs.keys())
            detection_year = None
            for year in years:
                if us_beliefs[year].p_covert_fab_exists >= threshold:
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

                # Track detection timing (for 0.5 threshold only)
                if threshold == 0.5:
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

                # Store individual values for 0.5 threshold (for dashboard)
                if threshold == 0.5:
                    individual_h100e_before_detection.append(h100e_at_detection)
                    individual_time_before_detection.append(operational_time)
                    individual_process_nodes.append(covert_fab.process_node.value)

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

                # Track never detected (for 0.5 threshold only)
                if threshold == 0.5:
                    never_detected_count += 1

                    # Check if never became operational
                    if final_year < operational_start:
                        never_operational_count += 1

                    # Debug: Check never-detected fabs with 0 compute
                    if h100e_at_end <= 0.0:
                        available_years = sorted(h100e_over_time.keys())
                        print(f"DEBUG: Never-detected fab with 0 compute - final_year={final_year}, operational_start={operational_start}, construction_start={construction_start}, duration={construction_duration}, h100e={h100e_at_end}, available_years={available_years}", flush=True)

                # Also store for dashboard (0.5 threshold only)
                if threshold == 0.5:
                    individual_h100e_before_detection.append(h100e_at_end)
                    individual_time_before_detection.append(operational_time)
                    individual_process_nodes.append(covert_fab.process_node.value)

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
    compute_ccdf = compute_ccdfs.get(0.5, [])
    op_time_ccdf = op_time_ccdfs.get(0.5, [])

    # Print USER'S DIRECT QUESTIONS FIRST
    print(f"\n=== DIRECT ANSWERS TO USER'S QUESTIONS ===", flush=True)
    print(f"1. Fabs that NEVER finished construction before simulation end: {fabs_never_finished_construction} ({100*fabs_never_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)
    print(f"2. Fabs that DID finish construction before simulation end: {fabs_finished_construction} ({100*fabs_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)
    print(f"3. Of those that finished, detected BEFORE finishing construction: {fabs_finished_and_detected_before_finish} ({100*fabs_finished_and_detected_before_finish/fabs_finished_construction:.1f}% of finished fabs)" if fabs_finished_construction > 0 else "3. Of those that finished, detected BEFORE finishing construction: 0", flush=True)
    print(f"==========================================\n", flush=True)

    # Print detection timing analysis
    print(f"\nDETECTION TIMING ANALYSIS (P(fab) >= 50%):", flush=True)
    print(f"  Total simulations with fab: {total_simulations_with_fab}", flush=True)
    print(f"  Detected during construction: {detected_during_construction_count} ({100*detected_during_construction_count/total_simulations_with_fab:.1f}%)", flush=True)
    print(f"  Detected after finished construction: {detected_after_operational_count} ({100*detected_after_operational_count/total_simulations_with_fab:.1f}%)", flush=True)
    print(f"  Never detected: {never_detected_count} ({100*never_detected_count/total_simulations_with_fab:.1f}%)", flush=True)
    print(f"  Fabs that never finished construction (should equal fabs never finished above): {fabs_never_finished_construction} ({100*fabs_never_finished_construction/total_simulations_with_fab:.1f}%)", flush=True)

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

    # Calculate statistics for LR components
    lr_components = {}
    if lr_inventory_by_sim:
        lr_inventory_array = np.array(lr_inventory_by_sim)
        lr_procurement_array = np.array(lr_procurement_by_sim)
        lr_other_array = np.array(lr_other_by_sim)

        # Analyze final timestep inventory LR distribution
        lr_inventory_final = [sim[-1] for sim in lr_inventory_by_sim]
        num_lr_equals_1 = sum(1 for lr in lr_inventory_final if abs(lr - 1.0) < 0.01)
        num_lr_gte_10 = sum(1 for lr in lr_inventory_final if lr >= 10.0)
        num_lr_gt_2_5 = sum(1 for lr in lr_inventory_final if lr > 2.5)
        total_sims = len(lr_inventory_final)
        print(f"INVENTORY LR ANALYSIS (final timestep):", flush=True)
        print(f"  Total simulations: {total_sims}", flush=True)
        print(f"  LR ≈ 1.0: {num_lr_equals_1} ({100*num_lr_equals_1/total_sims:.1f}%)", flush=True)
        print(f"  LR > 2.5: {num_lr_gt_2_5} ({100*num_lr_gt_2_5/total_sims:.1f}%)", flush=True)
        print(f"  LR ≥ 10.0: {num_lr_gte_10} ({100*num_lr_gte_10/total_sims:.1f}%)", flush=True)
        print(f"  Other values: {total_sims - num_lr_equals_1 - num_lr_gte_10} ({100*(total_sims - num_lr_equals_1 - num_lr_gte_10)/total_sims:.1f}%)", flush=True)

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
            "individual_us_probs": [list(sim) for sim in us_probs_by_sim],
            "individual_h100e": [list(np.array(sim) / 1e3) for sim in h100e_by_sim]
        },
        "compute_ccdf": compute_ccdf,
        "compute_ccdfs": compute_ccdfs,
        "op_time_ccdf": op_time_ccdf,
        "op_time_ccdfs": op_time_ccdfs,
        "lr_components": lr_components,
        "compute_factors": compute_factors,
        "architecture_efficiency_at_agreement": architecture_efficiency_at_agreement,
        "num_simulations": len(model.simulation_results),
        "individual_h100e_before_detection": individual_h100e_before_detection,
        "individual_time_before_detection": individual_time_before_detection,
        "individual_process_node": individual_process_nodes
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
