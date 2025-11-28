"""
Quick diagnostic script to test construction finish time variance.
"""
import numpy as np
from backend.model import Model
from backend.paramaters import ModelParameters, SimulationSettings, CovertProjectProperties, CovertProjectParameters

# Create parameters with small number of simulations for quick test
params = ModelParameters(
    simulation_settings=SimulationSettings(
        start_year=2030,
        end_year=2037,
        time_step_years=0.1,
        num_simulations=20
    ),
    covert_project_properties=CovertProjectProperties(),
    covert_project_parameters=CovertProjectParameters()
)

# Create and run model
print("Running 20 simulations...")
model = Model(params)
model.run_simulations(num_simulations=20)

# Extract construction finish times
construction_finish_times = []
for covert_projects, detectors in model.simulation_results:
    fab = covert_projects["prc_covert_project"].covert_fab
    if fab is not None:
        finish_time = fab.construction_start_year + fab.construction_duration
        construction_finish_times.append(finish_time)
        print(f"  Start: {fab.construction_start_year:.4f}, Duration: {fab.construction_duration:.4f}, Finish: {finish_time:.4f}")

if construction_finish_times:
    print(f"\nSummary:")
    print(f"  Min finish time: {min(construction_finish_times):.4f}")
    print(f"  Max finish time: {max(construction_finish_times):.4f}")
    print(f"  Range: {max(construction_finish_times) - min(construction_finish_times):.4f} years")
    print(f"  Median: {np.median(construction_finish_times):.4f}")
    print(f"  Std dev: {np.std(construction_finish_times):.4f}")

    # Check operational status at a few key years
    years_to_check = [2031.0, 2031.5, 2032.0, 2032.5, 2033.0]
    print(f"\nProportion operational at key years:")
    for year in years_to_check:
        count_operational = sum(1 for t in construction_finish_times if t <= year)
        proportion = count_operational / len(construction_finish_times)
        print(f"  Year {year}: {proportion:.2f} ({count_operational}/{len(construction_finish_times)} fabs)")
else:
    print("No fabs were built!")
