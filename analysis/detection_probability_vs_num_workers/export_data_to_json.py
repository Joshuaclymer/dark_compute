"""Export bayesian plot data and CSV data to JSON for frontend"""
import pandas as pd
import numpy as np
import pickle
import json

# Load the cached plot data from Monte Carlo simulations
with open('model_bayesian_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

# Extract the cached prediction data
x_pred = plot_data['X_plot_workers']
y_median = plot_data['y_plot']
y_low = plot_data['y_lower']
y_high = plot_data['y_upper']

# Load and prepare the actual data
df = pd.read_csv('nuclear_case_studies.csv')

def extract_years(text):
    """Extract years from 'Function identified' column"""
    if pd.isna(text) or 'NA' in str(text):
        return np.nan
    try:
        import re
        if 'several' in str(text).lower():
            return 6.0
        match = re.search(r'\(([<~]?\d+)\s*yr\)', str(text))
        if match:
            years_str = match.group(1)
            if '<' in years_str:
                return float(years_str.replace('<', '')) / 2
            if '~' in years_str:
                return float(years_str.replace('~', ''))
            return float(years_str)
    except:
        pass
    return np.nan

def extract_workers(text):
    """Extract numeric worker count"""
    if pd.isna(text):
        return np.nan
    try:
        import re
        text_clean = str(text).replace(',', '').replace('~', '')
        match = re.search(r'(\d+)', text_clean)
        if match:
            return float(match.group(1))
    except:
        pass
    return np.nan

df['detection_years'] = df['Function identified'].apply(extract_years)
df['num_workers'] = df['Workers (nuclear roles)'].apply(extract_workers)
df_clean = df.dropna(subset=['detection_years', 'num_workers'])

# Create the output data structure
output = {
    'prediction': {
        'x': x_pred.tolist(),
        'y_median': y_median.tolist(),
        'y_low': y_low.tolist(),
        'y_high': y_high.tolist()
    },
    'actual_data': {
        'x': df_clean['num_workers'].tolist(),
        'y': df_clean['detection_years'].tolist(),
        'sites': df_clean['Site'].tolist()
    }
}

# Write to JSON file in static directory
with open('../../static/detection_latency_data.json', 'w') as f:
    json.dump(output, f)

print("Data exported to ../../static/detection_latency_data.json")
