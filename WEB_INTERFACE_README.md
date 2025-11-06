# Covert Compute Production Model - Web Interface

This web interface allows you to interactively run the covert compute production model with customizable parameters.

## Features

### Interactive Plots

1. **Simulation Runs Over Time**
   - Shows US probability that a covert fab exists and H100e count over time
   - Displays median values with 25th-75th percentile bands
   - Shows up to 100 individual simulation runs with semi-transparent lines

2. **CCDF of Covert Compute Produced Before Detection**
   - Complementary cumulative distribution function showing the probability of producing at least X H100-equivalents before detection
   - Uses logarithmic scale for better visualization of the wide range

3. **CCDF of Operational Time Before Detection**
   - Shows the probability of operating for at least X years before detection

### Configurable Parameters

#### Simulation Settings
- **Agreement Start Year**: When the US-PRC agreement goes into force
- **End Year**: When to stop the simulation
- **Time Increment**: Simulation time step (in years)
- **Number of Simulations**: How many Monte Carlo runs to execute

#### PRC Strategy
- **Run Covert Project**: Whether PRC runs a covert project
- **Build Covert Fab**: Whether PRC builds a covert fab
- **Operating Labor**: Number of workers dedicated to fab operation
- **Construction Labor**: Number of workers dedicated to fab construction
- **Process Node**: Manufacturing process node (130nm, 28nm, 14nm, 7nm, or best available indigenously)
- **Scanner Proportion**: Fraction of PRC's domestic lithography scanners allocated to the fab

#### US Prior Probabilities
- **P(Project Exists)**: Initial US probability that a covert project exists
- **P(Fab Exists)**: Initial US probability that a covert fab exists

#### Fab Model Parameters
- **Wafers/Month per Worker**: Labor productivity (default: 24.64)
- **Wafers/Month per Scanner**: Scanner productivity (default: 1000)
- **Mean Detection Time (100 workers)**: Average detection time for 100 workers (default: 6.95 years)
- **Mean Detection Time (1000 workers)**: Average detection time for 1000 workers (default: 3.42 years)

## Running the Web Interface

1. Install Flask if not already installed:
   ```bash
   pip install flask
   ```

2. Start the Flask server:
   ```bash
   python3 app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

4. Adjust parameters in the sidebar and click "Run Simulation"

5. Wait for the simulation to complete (may take a few moments depending on the number of simulations)

6. Explore the interactive plots in the main area

## Technical Details

### Backend (app.py)
- Flask server that handles simulation requests
- Runs the Python model with specified parameters
- Extracts and formats data for visualization

### Frontend (templates/index.html)
- Responsive layout with sidebar for parameters and main area for plots
- Uses Plotly.js for interactive, publication-quality plots
- Real-time parameter updates with immediate visual feedback

## Notes

- The default parameters match those used in the original model
- Increasing the number of simulations will improve statistical accuracy but take longer to run
- The plots automatically adjust to the data range for optimal visualization
