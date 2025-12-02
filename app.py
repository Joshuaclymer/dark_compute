import os
import sys

# IMPORTANT: Change to the directory containing this file at startup.
# This prevents errors when the shell's working directory no longer exists
# (e.g., if a directory was deleted after a previous session).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_APP_DIR)

from flask import Flask, render_template, jsonify, request, send_file, Response
from backend.model import Model
from backend.paramaters import (
    CovertProjectProperties,
    CovertProjectParameters,
    SimulationSettings,
    ModelParameters,
    SlowdownPageParameters,
)
from backend.format_data_for_black_project_plots import extract_plot_data
from backend import util
import json
import hashlib
import math

app = Flask(__name__, template_folder='black_project_frontend')


def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization.

    Replaces float('inf'), float('-inf'), and float('nan') with None,
    since these are not valid JSON values.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    else:
        return obj

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

# Create a global ModelParameters instance that stays synchronized with the sidebar
app_params = ModelParameters(
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
    return send_file('slowdown_model_frontend/slowdown_model_v2.html')

@app.route('/slowdown_model_frontend/<path:filename>')
def serve_slowdown_files(filename):
    """Serve files from the slowdown_model_frontend directory."""
    return send_file(f'slowdown_model_frontend/{filename}')

@app.route('/slowdown_model_frontend/components/<path:filename>')
def serve_slowdown_component_files(filename):
    """Serve files from the slowdown_model_frontend/components directory."""
    return send_file(f'slowdown_model_frontend/components/{filename}')

@app.route('/<path:filename>')
def serve_html(filename):
    """Serve HTML, JS, CSS, and SVG files from the black_project_frontend directory."""
    # If the path already starts with 'black_project_frontend/', don't prepend it again
    if filename.startswith('black_project_frontend/'):
        filepath = filename
    else:
        filepath = f'black_project_frontend/{filename}'

    if filename.endswith('.html') and filename != 'index.html':
        return send_file(filepath)
    if filename.endswith('.js'):
        return send_file(filepath)
    if filename.endswith('.css'):
        return send_file(filepath)
    if filename.endswith('.svg'):
        return send_file(filepath, mimetype='image/svg+xml')
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

@app.route('/log_frontend', methods=['POST'])
def log_frontend():
    """Log frontend debug messages to the server console."""
    try:
        log_data = request.json
        logs = log_data.get('logs', [])
        print(f"\n{'='*80}", flush=True)
        print(f"FRONTEND LOGS ({len(logs)} messages):", flush=True)
        for log in logs:
            level = log.get('level', 'log')
            msg = log.get('message', '')
            data = log.get('data')
            if data:
                print(f"  [{level.upper()}] {msg}: {json.dumps(data, indent=2, default=str)}", flush=True)
            else:
                print(f"  [{level.upper()}] {msg}", flush=True)
        print(f"{'='*80}\n", flush=True)
        return jsonify({"status": "logged"}), 200
    except Exception as e:
        print(f"ERROR logging frontend: {e}", flush=True)
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

@app.route('/get_slowdown_model_data_stream')
def get_slowdown_model_data_stream():
    """Stream AI takeoff slowdown model data with progress updates via Server-Sent Events."""
    # Parse slowdown parameters from query arguments BEFORE entering the generator
    # (request context is not available inside the generator after the response starts)
    slowdown_params = SlowdownPageParameters()
    slowdown_params.update_from_dict(dict(request.args))

    def generate():
        import queue
        import threading

        progress_queue = queue.Queue()

        def progress_callback(current, total, trajectory_name):
            progress_queue.put({
                'type': 'progress',
                'current': current,
                'total': total,
                'trajectory': trajectory_name
            })

        def status_callback(status_message):
            progress_queue.put({
                'type': 'status',
                'message': status_message
            })

        def partial_data_callback(partial_data):
            # Sanitize and send partial data immediately
            sanitized = sanitize_for_json(partial_data)
            progress_queue.put({'type': 'partial', 'data': sanitized})

        def compute_data():
            try:
                from backend.format_data_for_slowdown_plots import get_slowdown_model_data_with_progress

                # Get cached simulation data
                status_callback("Loading cached covert compute data...")
                cached_simulation_data = load_default_cache()
                if not cached_simulation_data:
                    import glob as glob_module
                    cache_files = glob_module.glob(os.path.join(CACHE_DIR, '*.json'))
                    cache_files = [f for f in cache_files if not f.endswith('default.json')]
                    if cache_files:
                        most_recent = max(cache_files, key=os.path.getmtime)
                        try:
                            with open(most_recent, 'r') as f:
                                cache_data = json.load(f)
                            cached_simulation_data = cache_data.get('results', cache_data)
                        except Exception as e:
                            print(f"Error loading cache file: {e}", flush=True)

                result = get_slowdown_model_data_with_progress(
                    cached_simulation_data,
                    slowdown_params=slowdown_params,
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                    partial_data_callback=partial_data_callback
                )
                # Sanitize result to remove inf/nan values that aren't valid JSON
                sanitized_result = sanitize_for_json(result)
                progress_queue.put({'type': 'complete', 'data': sanitized_result})
            except Exception as e:
                import traceback
                traceback.print_exc()
                progress_queue.put({'type': 'error', 'error': str(e)})

        # Start computation in background thread
        thread = threading.Thread(target=compute_data)
        thread.start()

        # Stream progress updates
        while True:
            try:
                msg = progress_queue.get(timeout=60)  # 60 second timeout
                if msg['type'] == 'progress':
                    yield f"data: {json.dumps(msg)}\n\n"
                elif msg['type'] == 'status':
                    yield f"data: {json.dumps(msg)}\n\n"
                elif msg['type'] == 'partial':
                    # Send partial data immediately so frontend can render
                    yield f"data: {json.dumps(msg)}\n\n"
                elif msg['type'] == 'complete':
                    yield f"data: {json.dumps(msg)}\n\n"
                    break
                elif msg['type'] == 'error':
                    yield f"data: {json.dumps(msg)}\n\n"
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

        thread.join()

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })


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
    print(f"  start_agreement_at_specific_year: {data.get('simulation_settings.start_agreement_at_specific_year', 'NOT PROVIDED')}", flush=True)
    print(f"  num_years_to_simulate: {data.get('simulation_settings.num_years_to_simulate', 'NOT PROVIDED')}", flush=True)
    print(f"{'='*80}\n", flush=True)

    try:
        # Update app_params with values from request
        app_params.update_from_dict(data)

        # Debug: Log parameters after update
        print(f"\n{'='*80}", flush=True)
        print(f"PARAMETERS AFTER UPDATE:", flush=True)
        print(f"  num_simulations: {app_params.simulation_settings.num_simulations}", flush=True)
        print(f"  start_agreement_at_specific_year: {app_params.simulation_settings.start_agreement_at_specific_year}", flush=True)
        print(f"  num_years_to_simulate: {app_params.simulation_settings.num_years_to_simulate}", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Create model with ModelParameters object
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
