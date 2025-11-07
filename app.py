from flask import Flask, render_template, jsonify, request, send_file
from model import Model, CovertProjectStrategy
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
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json

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

    # US prior probabilities
    p_project_exists = float(data.get('p_project_exists', 0.2))
    p_fab_exists = float(data.get('p_fab_exists', 0.1))

    # Fab model parameters - update all parameters from data
    if 'proportion_of_diverted_sme_with_50p_chance_of_detection' in data:
        FabModelParameters.proportion_of_diverted_sme_with_50p_chance_of_detection = float(data['proportion_of_diverted_sme_with_50p_chance_of_detection'])
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

    # Create and run model
    model = Model(
        year_us_prc_agreement_goes_into_force=agreement_year,
        end_year=end_year,
        increment=increment
    )

    # Update initial detector beliefs
    model.initial_detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"][agreement_year].p_project_exists = p_project_exists
    model.initial_detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"][agreement_year].p_covert_fab_exists = p_fab_exists

    # Run simulations
    model.run_simulations(num_simulations=num_simulations)

    # Extract data for plots
    results = extract_plot_data(model)

    return jsonify(results)

def extract_plot_data(model):
    """Extract plot data from model simulation results"""
    if not model.simulation_results:
        return {"error": "No simulation results"}

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
                arch_efficiency = estimate_architecture_efficiency_relative_to_h100(year)
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
                h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                compute_at_detection.append(h100e_at_detection)

                construction_start = covert_fab.construction_start_year
                construction_duration = covert_fab.construction_duration
                operational_start = construction_start + construction_duration

                if detection_year >= operational_start:
                    operational_time = detection_year - operational_start
                else:
                    operational_time = 0.0

                operational_time_at_detection.append(operational_time)

                # Store individual values for 0.5 threshold (for dashboard)
                if threshold == 0.5:
                    individual_h100e_before_detection.append(h100e_at_detection)
                    individual_time_before_detection.append(operational_time)
                    individual_process_nodes.append(covert_fab.process_node.value)

        # Calculate CCDFs for this threshold
        if compute_at_detection:
            compute_sorted = np.sort(compute_at_detection)
            ccdf_compute = 1.0 - np.arange(1, len(compute_sorted) + 1) / len(compute_sorted)
            compute_ccdfs[threshold] = [{"x": float(x), "y": float(y)} for x, y in zip(compute_sorted, ccdf_compute)]

        if operational_time_at_detection:
            op_time_sorted = np.sort(operational_time_at_detection)
            ccdf_op_time = 1.0 - np.arange(1, len(op_time_sorted) + 1) / len(op_time_sorted)
            op_time_ccdfs[threshold] = [{"x": float(x), "y": float(y)} for x, y in zip(op_time_sorted, ccdf_op_time)]

    # Keep backward compatibility with single threshold
    compute_ccdf = compute_ccdfs.get(0.5, [])
    op_time_ccdf = op_time_ccdfs.get(0.5, [])

    # Calculate statistics for LR components
    lr_components = {}
    if lr_inventory_by_sim:
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
            "process_node_by_sim": process_node_by_sim
        }

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
