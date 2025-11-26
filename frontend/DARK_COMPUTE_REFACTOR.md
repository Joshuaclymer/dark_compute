# Dark Compute Model Section Refactor

## Overview
The large `dark_compute_model_section` files have been broken down into three smaller, more manageable files.

## Old Structure (Backed Up)
- `dark_compute_model_section.html.backup` - 27K (single large HTML file)
- `dark_compute_model_section.js.backup` - 89K (single large JS file with 1986 lines)

## New Structure

### 1. Top Section (Dashboard + Main Plots)
**Files:**
- `dark_compute_top_section.html` (4.1K)
- `dark_compute_top_section.js` (23K)

**Contains:**
- Dashboard with median outcome metrics
- H100-Years CCDF plot (computation performed before detection)
- AI R&D Reduction CCDF plot
- Simulation runs time series plot

**Key Functions:**
- `plotH100YearsTimeSeries(data)`
- `plotProjectH100YearsCcdf(data)`
- `plotAiRdReductionCcdf(data)`
- `updateDarkComputeModelDashboard(data)`
- `updateAiRdReduction(data, projectTimeMedian, projectH100YearsMedian)`

### 2. Rate of Computation Section
**Files:**
- `dark_compute_rate_section.html` (12K)
- `dark_compute_rate_section.js` (21K)

**Contains:**
- "How we estimate covert computation" section
- Chip stock breakdown (initial + acquired)
- Survival rate calculations
- Datacenter capacity limits
- Energy usage visualization
- Cumulative computation plots

**Key Functions:**
- `plotDarkComputeRateSection(data)`

**Plots Generated:**
- Initial dark compute (histogram)
- Covert fab flow (cumulative production)
- Chip survival rate
- Total dark compute stock
- Datacenter capacity
- Energy consumption (stacked area)
- Operational dark compute
- Covert computation (cumulative H100-years)

### 3. Detection Likelihood Section
**Files:**
- `dark_compute_detection_section.html` (10K)
- `dark_compute_detection_section.js` (42K)

**Contains:**
- "Predicting the likelihood of detection" section
- Intelligence evidence breakdown
- Historical detection data visualization

**Key Functions:**
- `plotDarkComputeDetectionSection(data)`
- `createDetectionLatencyPlot()`
- `createIntelligenceAccuracyPlot()`

**Plots Generated:**
- LR components for chip stock discrepancies
- LR components for SME discrepancies
- Combined evidence from reported assets
- Direct evidence of covert operations
- Posterior probability of covert project
- Historical detection latency (nuclear case studies)
- Intelligence accuracy estimates

### 4. Main Coordinator
**Files:**
- `dark_compute_main.js` (576B)

**Contains:**
- Main `plotDarkComputeModel(data)` function that orchestrates all three sections

## Integration
The `index.html` file now includes:
```html
<!-- HTML includes -->
{% include 'dark_compute_top_section.html' %}
{% include 'dark_compute_rate_section.html' %}
{% include 'dark_compute_detection_section.html' %}

<!-- Script includes -->
<script src="dark_compute_top_section.js"></script>
<script src="dark_compute_rate_section.js"></script>
<script src="dark_compute_detection_section.js"></script>
<script src="dark_compute_main.js"></script>
```

## Benefits
1. **Easier Navigation**: Each file is focused on a specific section (200-600 lines vs 2000 lines)
2. **Better Organization**: HTML and JS for each section are clearly paired
3. **Easier Maintenance**: Changes to one section don't require scrolling through all code
4. **Logical Structure**: File division matches the visual sections on the page
5. **Reduced Cognitive Load**: Developers can focus on one section at a time

## Testing
After this refactor:
1. Test that all plots render correctly
2. Verify dashboard updates work
3. Check that parameter click handlers still function
4. Ensure plot resizing works properly
5. Test with both live simulations and cached data

## Rollback
If issues occur, the original files are backed up:
- `dark_compute_model_section.html.backup`
- `dark_compute_model_section.js.backup`

To rollback:
```bash
cd frontend
mv dark_compute_model_section.html.backup dark_compute_model_section.html
mv dark_compute_model_section.js.backup dark_compute_model_section.js
```

Then update `index.html` to use the original includes and scripts.
