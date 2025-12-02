#!/usr/bin/env python3
"""
Script to generate default cache for the web application.
Run this with: python3 generate_default_cache.py [num_simulations]
"""

import sys
import json
from black_project_backend.model import Model
from black_project_backend.black_project_parameters import ModelParameters, SimulationSettings, CovertProjectProperties, CovertProjectParameters
from black_project_backend.format_data_for_black_project_plots import extract_plot_data
import os

# Cache directory and save function (copied from app.py to avoid circular imports)
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def save_default_cache(results):
    """Save results as the default cache."""
    cache_file = os.path.join(CACHE_DIR, 'default.json')
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    print(f"Saved default cache: {cache_file}", flush=True)

def generate_cache(num_simulations=10):
    """Generate and save default cache with specified number of simulations."""
    print(f"Generating cache with {num_simulations} simulations...")

    # Create default parameters
    sim_settings = SimulationSettings(num_simulations=num_simulations)
    app_params = ModelParameters(
        simulation_settings=sim_settings,
        covert_project_properties=CovertProjectProperties(),
        covert_project_parameters=CovertProjectParameters()
    )

    # Create and run model
    print("Creating model...")
    model = Model(app_params)

    print("Running simulations...")
    model.run_simulations(num_simulations=num_simulations)

    print("Extracting plot data...")
    results = extract_plot_data(model, app_params)

    print("Saving to cache...")
    save_default_cache(results)

    print(f"✓ Successfully generated default cache with {num_simulations} simulations")
    return results

if __name__ == '__main__':
    num_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    generate_cache(num_sims)
