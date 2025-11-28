# Quick Start: Predicting AI Milestones from Compute

## TL;DR - Simplest Example

```python
import numpy as np
from predict_trajectory import predict_milestones_from_total_compute, print_milestone_summary

# You have 100,000 H100s - just specify total compute!
years = np.linspace(2024, 2045, 200)
total_compute = 1e5 * 2**((years - 2024) / 2.0)  # Doubling every 2 years

# Predict when AI milestones happen
milestones = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute
)

# See results
print_milestone_summary(milestones)
```

That's it! No need to manually split between inference and experiment compute.

## What You Get

The model predicts when these AI capability milestones are achieved:

- **AC** (Automated Coder) - AI can automate software engineering
- **AI2027-SC** (Superhuman Coder) - AI coding exceeds human capability
- **AIR-5x through AIR-10000x** - AI accelerates R&D by 5x to 10,000x
- **SAR** (Superhuman AI Researcher) - Top 0.1% experiment selection
- **SIAR** (Significantly Improved AI Researcher) - 3x beyond SAR
- **STRAT-AI** (Strategic AI) - Strategically aware systems
- **TED-AI** (Transformative Economic Disruption) - Economy-transforming AI
- **ASI** (Artificial Superintelligence) - Far beyond human capability

## What About Inference vs Experiment Compute?

**You asked: "If I have 100,000 H100s, do I need to decide how to split them?"**

**Answer: No!** Just use `predict_milestones_from_total_compute()` and it automatically:
- Allocates **15% to inference** (for AI coding assistants)
- Allocates **85% to experiments** (for actual training/testing)

**Why this works:** The split ratio has minimal impact on results. Testing shows:
- Varying inference from 5% to 20% only changes ASI timing by ~3 weeks
- The model is robust to different allocation strategies
- Default 15%/85% matches historical data

## Advanced: If You Want Control

### Custom Split Ratio
```python
# Allocate only 10% to inference, 90% to experiments
milestones = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute,
    inference_fraction=0.10
)
```

### Manual Split
```python
from predict_trajectory import predict_milestones_from_compute

# Manually specify both
inference_compute = 0.15 * total_compute
experiment_compute = 0.85 * total_compute

milestones = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_compute,
    experiment_compute=experiment_compute
)
```

### Optimal Split (Most Sophisticated)
```python
from compute_allocation import optimal_compute_split

# Find the mathematically optimal split
inference, experiment, max_rate = optimal_compute_split(
    total_compute=100000.0,
    L_HUMAN=100.0,
    automation_fraction=0.3,
    research_stock=5000.0
)

print(f"Optimal: {inference/100000:.1%} inference, {experiment/100000:.1%} experiment")
```

**Note:** The optimizer typically finds 95-99% should go to experiments with large compute budgets.

## Run the Examples

Three example files demonstrate different use cases:

```bash
# Basic examples with different compute trajectories
python3 example_usage.py

# Total compute examples (automatic splitting)
python3 example_total_compute.py

# Optimization examples
python3 compute_allocation.py
```

## More Information

- **Full Documentation**: See `README_TRAJECTORY_PREDICTION.md`
- **All Files**:
  - `predict_trajectory.py` - Main prediction module
  - `compute_allocation.py` - Optimal allocation tools
  - `example_*.py` - Working examples
