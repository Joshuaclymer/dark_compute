# Add Weight Stealing and Algorithm Stealing Features

## Summary

This PR adds two new features to the AI futures calculator that model scenarios where one project steals from a more capable leading project:

1. **Weight Stealing**: At specified times/milestones, the stealing project gains access to the leading project's AI capabilities (automation fraction, AI research taste) but keeps its own progress and research stock.

2. **Algorithm Stealing**: Up to a specified time/milestone, the stealing project has the same software efficiency as the leading project, modeling scenarios where algorithms and techniques are stolen.

Both features can be combined in a single simulation.

## API Changes

### New Parameters

**`TrajectoryPredictor.predict_from_time_series()`** and **`predict_milestones_from_compute()`** now accept:

- `inference_compute_leading_project`: AI inference compute for the leading project
- `experiment_compute_leading_project`: Experiment compute for the leading project
- `L_HUMAN_leading_project`: Human labor for the leading project
- `years_weights_are_stolen_from_leading_project`: List of times or milestone names (e.g., `[2030.0, "SC"]`) when weights are stolen
- `stealing_algorithms_up_to`: Time or milestone up to which algorithms are stolen (e.g., `"SC"` or `2035.0`)

### Usage Examples

```python
from predict_trajectory import predict_milestones_from_compute
import numpy as np

time = np.linspace(2024, 2050, 100)

# Stealing project (smaller)
inference = 1e5 * np.exp(0.3 * (time - 2024))
experiment = 1e4 * np.exp(0.3 * (time - 2024))

# Leading project (larger)
inference_leader = 1e6 * np.exp(0.5 * (time - 2024))
experiment_leader = 1e5 * np.exp(0.4 * (time - 2024))

# Weight stealing at SC milestone
milestones = predict_milestones_from_compute(
    time=time,
    inference_compute=inference,
    experiment_compute=experiment,
    inference_compute_leading_project=inference_leader,
    experiment_compute_leading_project=experiment_leader,
    years_weights_are_stolen_from_leading_project=["SC"]
)

# Algorithm stealing up to SC milestone
milestones = predict_milestones_from_compute(
    time=time,
    inference_compute=inference,
    experiment_compute=experiment,
    inference_compute_leading_project=inference_leader,
    experiment_compute_leading_project=experiment_leader,
    stealing_algorithms_up_to="SC"
)

# Combined: both weight and algorithm stealing
milestones = predict_milestones_from_compute(
    time=time,
    inference_compute=inference,
    experiment_compute=experiment,
    inference_compute_leading_project=inference_leader,
    experiment_compute_leading_project=experiment_leader,
    years_weights_are_stolen_from_leading_project=["SC"],
    stealing_algorithms_up_to=2030.0
)
```

## Implementation Details

### Weight Stealing Semantics

When weights are stolen at time T:
- The stealing project gains access to the leading project's **model capabilities** at time T
- This means higher automation fraction and better AI research taste
- The stealing project keeps its own progress and research stock
- Future progress is accelerated because the AI assistant is more capable

### Algorithm Stealing Semantics

When algorithms are stolen up to time T:
- The stealing project has the same **software efficiency** as the leading project for all times ≤ T
- Software efficiency = progress - training_compute (the algorithmic improvements)
- After time T, the stealing project continues with its own trajectory but keeps the accumulated gains

### Milestone Name Resolution

Times can be specified as:
- Numeric values (decimal years): `2030.5`
- Milestone names: `"SC"`, `"AC"`, `"ASI"`, `"AI2027-SC"`, etc.

Aliases are supported:
- `"SC"` → `"AI2027-SC"`
- `"SAR"` → `"SAR-level-experiment-selection-skill"`
- `"SIAR"` → `"SIAR-level-experiment-selection-skill"`

## Files Changed

- `progress_model.py`: Added `WeightStealingProgressModel` class and new parameters
- `predict_trajectory.py`: Updated API to accept stealing parameters
- `model_config.py`: Added default parameter values
- `test_weight_stealing.py`: Comprehensive test suite (15 tests)

## Testing

All 15 tests pass:
- Basic weight stealing functionality
- Milestone name resolution
- Algorithm stealing
- Combined weight + algorithm stealing
- Edge cases (stealing at start/end, multiple events)

Backward compatibility verified - existing API works unchanged when no stealing parameters are provided.

## Known Limitations

1. **Algorithm stealing discontinuity**: The current implementation applies a software efficiency boost up to the cutoff time. After the cutoff, the boost no longer applies to new time points. This is technically correct but may result in unexpected behavior at the boundary.

2. **Performance**: The ODE integration creates closures that capture large arrays. This is efficient enough for typical use but could be optimized further if needed.
