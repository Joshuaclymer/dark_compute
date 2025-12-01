# Project-Specific Instructions

## Flask Server Configuration

Use port 5001 when running the Flask application (default, avoiding macOS AirPlay on port 5000).

When starting the Flask server, use:
```bash
cd /Users/joshuaclymer/github/covert_compute_production_model && python3 app.py
```

The app is configured to run on http://127.0.0.1:5001

Use existing functions and ccs styles when you can, and keep code clean.

If there's an issue related to a plot not showing up, probably it's because the height of the plot was not set in the CSS rules. 

Put logic on the backend instead of the front-end. Just display data on the front-end with plots.

## Code Organization

**IMPORTANT**: Always add core logic to `backend/classes/`. The files in `backend/serve_slowdown_model/` should only be used for:
- Formatting data for plots
- Orchestrating calls to classes
- Building API responses

Core logic such as computations, algorithms, and business rules must be implemented in the appropriate class files under `backend/classes/`.
