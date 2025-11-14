# Project-Specific Instructions

## Flask Server Configuration

Use port 5000 when running the Flask application (default).

When starting the Flask server, use:
```bash
cd /Users/joshuaclymer/github/covert_compute_production_model && lsof -ti:5000 2>/dev/null | xargs kill -9 2>/dev/null ; python3 app.py
```

The app is configured to run on http://127.0.0.1:5000
