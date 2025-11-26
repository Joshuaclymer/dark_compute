from flask import Flask, render_template, jsonify, request, send_file
from backend.model import Model
from backend.paramaters import (
    CovertProjectProperties,
    CovertProjectParameters,
    SimulationSettings,
    Parameters
)
from backend.serve_data_for_dark_compute_model import extract_plot_data
from backend import util
import json
import os
import hashlib

app = Flask(__name__, template_folder='frontend')

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(params_dict):
    """Generate a cache key from parameters dictionary."""
    # Convert dict to sorted JSON string for consistent hashing
    params_json = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_json.encode()).hexdigest()

def save_cache(params_dict, results):
    """Save simulation results to cache."""
    cache_key = get_cache_key(params_dict)
    cache_file = os.path.join(CACHE_DIR, f'{cache_key}.json')
    with open(cache_file, 'w') as f:
        json.dump({'params': params_dict, 'results': results}, f)
    print(f"Saved cache: {cache_file}", flush=True)
    return cache_key

def load_cache(params_dict):
    """Load simulation results from cache if available."""
    cache_key = get_cache_key(params_dict)
    cache_file = os.path.join(CACHE_DIR, f'{cache_key}.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded cache: {cache_file}", flush=True)
        return data['results']
    return None

def get_default_cache_file():
    """Get the path to the default cache file."""
    return os.path.join(CACHE_DIR, 'default.json')

def save_default_cache(results):
    """Save results as the default cache."""
    cache_file = get_default_cache_file()
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    print(f"Saved default cache: {cache_file}", flush=True)

def load_default_cache():
    """Load the default cache if available."""
    cache_file = get_default_cache_file()
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded default cache: {cache_file}", flush=True)
        return results
    return None

# Create a global Parameters instance that stays synchronized with the sidebar
app_params = Parameters(
    simulation_settings=SimulationSettings(),
    covert_project_properties=CovertProjectProperties(),
    covert_project_parameters=CovertProjectParameters()
)

@app.route('/')
def index():
    # Pass current parameter values to template
    defaults = app_params.to_dict()
    return render_template('index.html', defaults=defaults)

@app.route('/slowdown')
def slowdown():
    """Serve the AI development slowdown model page."""
    return send_file('slowdown_model_frontend/slowdown_model.html')

@app.route('/slowdown_model_frontend/<path:filename>')
def serve_slowdown_files(filename):
    """Serve files from the slowdown_model_frontend directory."""
    return send_file(f'slowdown_model_frontend/{filename}')

@app.route('/<path:filename>')
def serve_html(filename):
    """Serve HTML, JS, CSS, and SVG files from the frontend directory."""
    if filename.endswith('.html') and filename != 'index.html':
        return send_file(f'frontend/{filename}')
    if filename.endswith('.js'):
        return send_file(f'frontend/{filename}')
    if filename.endswith('.css'):
        return send_file(f'frontend/{filename}')
    if filename.endswith('.svg'):
        return send_file(f'frontend/{filename}', mimetype='image/svg+xml')
    return '', 404

@app.route('/log_client_error', methods=['POST'])
def log_client_error():
    """Log JavaScript errors from the client to the server console."""
    try:
        error_data = request.json
        print(f"\n{'='*80}", flush=True)
        print(f"JAVASCRIPT ERROR:", flush=True)
        print(f"  Message: {error_data.get('message', 'No message')}", flush=True)
        if 'source' in error_data:
            print(f"  Source: {error_data.get('source')}:{error_data.get('lineno')}:{error_data.get('colno')}", flush=True)
        if 'reason' in error_data:
            print(f"  Reason: {error_data.get('reason')}", flush=True)
        print(f"  Stack: {error_data.get('stack', 'No stack trace')}", flush=True)
        print(f"{'='*80}\n", flush=True)
        return jsonify({"status": "logged"}), 200
    except Exception as e:
        print(f"ERROR logging client error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get_default_results')
def get_default_results():
    """Get cached default simulation results."""
    try:
        results = load_default_cache()
        if results:
            return jsonify(results)
        else:
            return jsonify({"error": "No default cache available"}), 404
    except Exception as e:
        print(f"ERROR loading default cache: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_slowdown_model_data')
def get_slowdown_model_data():
    """Get AI takeoff slowdown model data with default parameters."""
    try:
        from backend.serve_data_for_slowdown_model import extract_takeoff_slowdown_trajectories

        # Use default parameters
        params = Parameters(
            simulation_settings=SimulationSettings(),
            covert_project_properties=CovertProjectProperties(),
            covert_project_parameters=CovertProjectParameters()
        )

        # Extract takeoff trajectories
        trajectories = extract_takeoff_slowdown_trajectories(params)

        return jsonify({
            'takeoff_trajectories': trajectories,
            'agreement_year': params.simulation_settings.start_year
        })
    except Exception as e:
        print(f"ERROR computing slowdown model data: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
    except Exception as e:
        print(f"ERROR parsing request: {e}", flush=True)
        return jsonify({"error": str(e)}), 400

    # Check if we should try to use cache
    use_cache = data.get('use_cache', True)

    # Try to load from cache first if enabled
    if use_cache:
        cached_results = load_cache(data)
        if cached_results:
            print(f"Using cached results", flush=True)
            return jsonify(cached_results)

    # Clear caches
    util._cache.clear()

    # Debug: Log received parameters
    print(f"\n{'='*80}", flush=True)
    print(f"RECEIVED PARAMETERS:", flush=True)
    print(f"  num_simulations: {data.get('simulation_settings.num_simulations', 'NOT PROVIDED')}", flush=True)
    print(f"  start_year: {data.get('simulation_settings.start_year', 'NOT PROVIDED')}", flush=True)
    print(f"  end_year: {data.get('simulation_settings.end_year', 'NOT PROVIDED')}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Update app_params with values from request
    app_params.update_from_dict(data)

    # Debug: Log parameters after update
    print(f"\n{'='*80}", flush=True)
    print(f"PARAMETERS AFTER UPDATE:", flush=True)
    print(f"  num_simulations: {app_params.simulation_settings.num_simulations}", flush=True)
    print(f"  start_year: {app_params.simulation_settings.start_year}", flush=True)
    print(f"  end_year: {app_params.simulation_settings.end_year}", flush=True)
    print(f"{'='*80}\n", flush=True)

    try:
        # Create model with Parameters object
        model = Model(app_params)

        # Run simulations
        model.run_simulations(num_simulations=app_params.simulation_settings.num_simulations)

        # Extract data for plots (includes initial stock data and type conversion)
        results = extract_plot_data(model, app_params)

        # Save to cache
        save_cache(data, results)

        return jsonify(results)

    except Exception as e:
        print(f"ERROR running simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

@app.route('/download/data_center_workers_vs_build_rate')
def download_data_center_workers_vs_build_rate():
    """Download the data center workers vs build rate CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'data_center_workers_vs_build_rate.csv')
    return send_file(file_path, as_attachment=True, download_name='data_center_workers_vs_build_rate.csv')

if __name__ == '__main__':
    import sys
    import os
    port = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else 5001))
    app.run(debug=True, port=port)
