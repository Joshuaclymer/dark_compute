# Trajectory Prediction Module

This module provides a simple interface to predict AI capability milestones from arbitrary compute time series.

## Files

- **`predict_trajectory.py`** - Main module with prediction functions and classes
- **`example_usage.py`** - Comprehensive examples demonstrating different use cases

## Quick Start

### Simplest Usage: Total Compute (Recommended)

```python
import numpy as np
from predict_trajectory import predict_milestones_from_total_compute, print_milestone_summary

# Define total compute trajectory (no need to split!)
years = np.linspace(2024, 2045, 200)
total_compute = 1e5 * 2**((years - 2024) / 2.0)  # Doubling every 2 years

# Predict milestones - automatic split between inference and experiment
milestones = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute
)

# Print summary
print_milestone_summary(milestones)
```

**Note:** The system automatically allocates 15% to inference (coding automation) and 85% to experiments by default. The split ratio has minimal impact on milestone timing (~weeks of difference).

### Advanced Usage: Manual Split

If you want control over the inference/experiment split:

```python
from predict_trajectory import predict_milestones_from_compute, print_milestone_summary

# Define compute trajectory with manual split
years = np.linspace(2024, 2045, 200)
inference_compute = 1e6 * 2**((years - 2024) / 2.0)  # For coding automation
experiment_compute = 1e5 * 2**((years - 2024) / 2.0)  # For running experiments

# Predict milestones
milestones = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_compute,
    experiment_compute=experiment_compute
)

# Print summary
print_milestone_summary(milestones)
```

### Load from CSV

```python
from predict_trajectory import predict_trajectory_from_csv

milestones, trajectory = predict_trajectory_from_csv('input_data.csv')
```

## Capability Milestones

The module predicts when the following AI capability milestones are achieved:

### Core Milestones
- **AC** - Automated Coder: AI can automate software engineering tasks
- **AI2027-SC** - Superhuman Coder: AI coding exceeds human parallel capability
- **SAR** - Superhuman AI Researcher: Top 0.1% experiment selection skill
- **SIAR** - Significantly Improved AI Researcher: 3x SAR level
- **STRAT-AI** - Strategic AI: Strategically aware AI system
- **TED-AI** - Transformative Economic Disruption AI
- **ASI** - Artificial Superintelligence: Far beyond human capability

### AI R&D Acceleration Milestones
- **AIR-5x** - AI accelerates software R&D by 5x
- **AIR-25x** - AI accelerates software R&D by 25x
- **AIR-250x** - AI accelerates software R&D by 250x
- **AIR-2000x** - AI accelerates software R&D by 2000x
- **AIR-10000x** - AI accelerates software R&D by 10000x

## Compute Allocation

### Understanding Inference vs Experiment Compute

The model distinguishes between two types of compute:

**Inference Compute:**
- Used for AI-powered coding automation (e.g., GitHub Copilot, AI coding assistants)
- Helps AI systems write code for experiments
- Combined with human labor to produce "coding labor"
- Typically **10-20%** of total compute

**Experiment Compute:**
- Used for running actual AI experiments and training runs
- The compute budget for training models, running evaluations
- Combined with coding labor to produce "experiment capacity"
- Typically **80-90%** of total compute

### Automatic Allocation

If you don't want to manually split compute, use `predict_milestones_from_total_compute()`:

```python
# Just specify total - automatic 15%/85% split
milestones = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute_array
)

# Or customize the split
milestones = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute_array,
    inference_fraction=0.10  # 10% inference, 90% experiment
)
```

**Key Finding:** The exact split ratio has minimal impact on milestone timing. Testing shows that varying inference from 5% to 20% only changes ASI timing by ~3 weeks. This is because both types of compute contribute to progress through the nested CES production functions.

### Optimal Allocation (Advanced)

For advanced users, `compute_allocation.py` provides optimization tools:

```python
from compute_allocation import optimal_compute_split

# Find optimal split for a single point
inference, experiment, rate = optimal_compute_split(
    total_compute=100000.0,
    L_HUMAN=100.0,
    automation_fraction=0.3,
    research_stock=5000.0
)
```

The optimizer typically finds that **most compute should go to experiments** (95-99%) when you have large compute budgets and moderate automation.

## API Reference

### Main Functions

#### `predict_milestones_from_total_compute()` ⭐ **Recommended**

Simplest function - predict milestones from total compute (automatic split).

**Parameters:**
- `time` (array): Time points in decimal years (e.g., 2024.0, 2024.5, ...)
- `total_compute` (array): Total compute budget at each time (H100-years)
- `L_HUMAN` (array, optional): Human labor supply (defaults to 100)
- `training_compute_growth_rate` (array, optional): Training compute growth rate in OOMs/year (defaults to 0.5)
- `initial_progress` (float, optional): Starting progress level in OOMs (default: 0.0)
- `params` (Parameters, optional): Custom model parameters (uses defaults if None)
- `inference_fraction` (float, optional): Fraction of total for inference (default: 0.15)

**Returns:** Dictionary mapping milestone names to `MilestoneInfo` objects.

#### `predict_milestones_from_compute()`

Advanced function - predict milestones with manual inference/experiment split.

**Parameters:**
- `time` (array): Time points in decimal years (e.g., 2024.0, 2024.5, ...)
- `inference_compute` (array): AI inference compute at each time (H100-years)
- `experiment_compute` (array): Experiment compute budget at each time (H100-years)
- `L_HUMAN` (array, optional): Human labor supply (defaults to 100)
- `training_compute_growth_rate` (array, optional): Training compute growth rate in OOMs/year (defaults to 0.5)
- `initial_progress` (float, optional): Starting progress level in OOMs (default: 0.0)
- `params` (Parameters, optional): Custom model parameters (uses defaults if None)

**Returns:** Dictionary mapping milestone names to `MilestoneInfo` objects.

#### `predict_trajectory_from_csv()`

Load time series from CSV and predict milestones.

**Parameters:**
- `csv_path` (str): Path to CSV with columns: time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate
- `params` (Parameters, optional): Model parameters
- `initial_progress` (float, optional): Starting progress level

**Returns:** Tuple of (milestones dict, full trajectory dict)

#### `print_milestone_summary()`

Print a formatted summary of milestone predictions.

**Parameters:**
- `milestones` (dict): Dictionary of milestone predictions

### Classes

#### `MilestoneInfo`

Dataclass containing information about a capability milestone:
- `time` - When milestone is achieved (decimal years)
- `progress_level` - Progress in OOMs of effective compute
- `research_effort` - Research effort at milestone
- `research_stock` - Cumulative research stock at milestone
- `progress_multiplier` - AI R&D speedup multiplier
- `metric_name` - Which metric crossed threshold
- `target_value` - Threshold value that was crossed

#### `TrajectoryPredictor`

Class for predicting trajectories with full access to intermediate results.

**Methods:**
- `predict_from_time_series()` - Predict from time series arrays
- `get_full_trajectory()` - Get complete trajectory with all metrics
- `get_milestone_times()` - Get just milestone achievement times

## Input Data Format

Time series can be provided as:

1. **NumPy arrays** - Direct input to `predict_milestones_from_compute()`
2. **CSV file** - Load with `predict_trajectory_from_csv()`

CSV format (5 columns):
```csv
time,L_HUMAN,inference_compute,experiment_compute,training_compute_growth_rate
2024.0,100.0,1000000.0,100000.0,0.5
2025.0,105.0,1500000.0,150000.0,0.5
...
```

Where:
- `time` - Decimal years
- `L_HUMAN` - Human labor supply (software engineers)
- `inference_compute` - AI inference compute (H100-year equivalents)
- `experiment_compute` - Experiment compute budget (H100-year equivalents)
- `training_compute_growth_rate` - Training compute growth rate (OOMs/year)

## Output Trajectory Metrics

When using `TrajectoryPredictor.get_full_trajectory()`, you get access to many metrics:

**Core Trajectories:**
- `times` - Time array
- `progress` - Cumulative progress (OOMs)
- `progress_rates` - Progress rate (OOMs/year)
- `research_stock` - Cumulative research effort
- `research_efforts` - Research effort per time

**AI Capabilities:**
- `automation_fraction` - Fraction of SWE tasks automated
- `ai_research_taste` - AI experiment selection skill
- `ai_research_taste_sd` - AI taste in standard deviations
- `ai_coding_labor_multipliers` - AI coding labor per human
- `ai_sw_progress_mult_ref_present_day` - AI R&D acceleration

...and many more (see full list in trajectory dict keys).

## Examples

See `example_usage.py` for comprehensive examples including:

1. Simple exponential growth
2. Slower growth trajectory
3. Compute growth with slowdown
4. Accessing full trajectory metrics
5. Loading from CSV file
6. Comparing multiple scenarios

Run examples:
```bash
python3 example_usage.py
```

## Notes

- Compute values are in **H100-year equivalents** (1 H100-year = 1 H100 GPU running for 1 year)
- Progress is measured in **OOMs (orders of magnitude) of effective compute**
- Times are in **decimal years** (2024.5 = mid-2024)
- Milestones are only included in results if achieved within the time range
- The model uses CES (Constant Elasticity of Substitution) production functions
- Human labor defaults to 100 software engineers if not specified
- Training compute growth defaults to 0.5 OOMs/year if not specified

## Integration with Existing Code

This module uses the same underlying `ProgressModel` as the web application (`app.py`), ensuring consistency with:
- Web UI predictions
- Monte Carlo simulations (`scripts/batch_rollout.py`)
- Analysis scripts (`scripts/plot_*.py`)

All parameters and configurations from `model_config.py` and `Parameters` class are fully supported.
