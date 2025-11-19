import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Read the workers vs build rate data (contains both workers and GW/year)
workers_df = pd.read_csv('data_center_workers_vs_build_rate.csv')

# Read the timelines data (for construction type lookup)
df = pd.read_csv('data_centers/data_center_timelines.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create a mapping from workers CSV names to timeline names
name_mapping = {
    'Amazon Canton Mississippi (Project Atlas)': 'Amazon Canton Mississippi',
    'Amazon Ridgeland Mississippi (Project Atlas)': 'Amazon Ridgeland Mississippi',
    'Anthropic-Amazon Project Rainier New Carlisle Indiana': 'Anthropic-Amazon Project Rainier New Carlisle Indiana',
    'Crusoe Goodnight Texas (Goodnight AI Campus)': 'Crusoe Goodnight Texas',
    'Google New Albany Ohio': 'Google New Albany Ohio',
    'Meta Hyperion Holly Ridge Louisiana': 'Meta Hyperion Holly Ridge Louisiana',
    'Meta Prometheus New Albany Ohio': 'Meta Prometheus New Albany Ohio',
    'Microsoft Fairwater Mount Pleasant Wisconsin': 'Microsoft Fairwater Mount Pleasant Wisconsin',
    'Microsoft Project Firecracker Rome Georgia': 'Microsoft Fayetteville Georgia',
    'OpenAI-Oracle Stargate Abilene Texas': 'OpenAI-Oracle Stargate Abilene Texas',
    'xAI Colossus 1 Memphis Tennessee': 'xAI Colossus 1 Memphis Tennessee',
    'xAI Colossus 2 Memphis Tennessee': 'xAI Colossus 2 Memphis Tennessee',
    'ORNL Frontier Supercomputer Oak Ridge Tennessee': 'ORNL Frontier Supercomputer Oak Ridge Tennessee',
    'LLNL El Capitan Phase 2 Livermore California': 'LLNL El Capitan Phase 2 Livermore California'
}

# Build results from the CSV
results = []

for idx, row in workers_df.iterrows():
    worker_name = row['Data Center Name']
    worker_count = row['Number of Concurrent Workers']
    gw_per_year = row['Gigawatts per Year']

    # Skip if no data
    if pd.isna(worker_count) or pd.isna(gw_per_year):
        continue
    if worker_count == "Can't find data" or worker_count == "No specific data available" or worker_count == "":
        continue
    if gw_per_year == "Can't find data" or gw_per_year == "" or gw_per_year == "No specific data available":
        continue

    try:
        # Handle approximate values like "~700"
        worker_count_str = str(worker_count).replace('~', '').replace(',', '').strip()
        worker_count = int(worker_count_str)
        gw_per_year = float(gw_per_year)
        # Convert GW/year to MW/year
        mw_per_year = gw_per_year * 1000
    except (ValueError, TypeError):
        continue

    # Get timeline name from mapping
    timeline_name = name_mapping.get(worker_name, worker_name)

    # Get timeline data to determine construction type
    dc_data = df[df['Data center'] == timeline_name].copy()

    # Determine construction type
    construction_type = 'Traditional'
    if len(dc_data) > 0:
        if 'xAI Colossus' in timeline_name:
            construction_type = 'Retrofit'
        elif 'Meta Prometheus' in timeline_name or 'Meta Hyperion' in timeline_name:
            # Check for tents in construction status
            tent_mentions = dc_data[dc_data['Construction status'].str.contains('tent', case=False, na=False)]
            if len(tent_mentions) > 0:
                construction_type = 'Tent'

    results.append({
        'Data Center': timeline_name,
        'Workers': worker_count,
        'MW per Year': mw_per_year,
        'Construction Type': construction_type
    })

# Create dataframe
results_df = pd.DataFrame(results)

print("Data Centers with both worker and timeline data:")
print(results_df.to_string(index=False))

# Create scatter plot
fig = go.Figure()

# Define colors and symbols for construction types
construction_colors = {
    'Traditional': 'blue',
    'Tent': 'orange',
    'Retrofit': 'green'
}

construction_symbols = {
    'Traditional': 'circle',
    'Tent': 'square',
    'Retrofit': 'diamond'
}

# Shorten datacenter names for labels
def shorten_name(name):
    if 'Amazon Ridgeland' in name:
        return 'Amazon Ridgeland'
    elif 'Amazon Canton' in name:
        return 'Amazon Canton'
    elif 'Anthropic-Amazon' in name:
        return 'Anthropic-Amazon Indiana'
    elif 'Google New Albany' in name:
        return 'Google New Albany'
    elif 'Meta Hyperion' in name:
        return 'Meta Hyperion'
    elif 'Meta Prometheus' in name:
        return 'Meta Prometheus'
    elif 'Microsoft Fairwater' in name:
        return 'Microsoft Fairwater'
    elif 'Microsoft' in name and 'Georgia' in name:
        return 'Microsoft Georgia'
    elif 'OpenAI-Oracle Stargate' in name:
        return 'Stargate Abilene'
    elif 'xAI Colossus 1' in name:
        return 'Colossus 1 Memphis'
    elif 'xAI Colossus 2' in name:
        return 'Colossus 2 Memphis'
    elif 'ORNL Frontier' in name:
        return 'Frontier Oak Ridge'
    elif 'LLNL El Capitan' in name:
        return 'El Capitan'
    else:
        return name

# Add scatter points individually with custom positioning
for idx, row in results_df.iterrows():
    datacenter = row['Data Center']
    short_name = shorten_name(datacenter)
    workers = row['Workers']
    mw_per_year = row['MW per Year']
    const_type = row['Construction Type']

    # Determine text position and alignment based on datacenter
    if 'LLNL El Capitan' in datacenter:
        # Bottom left: anchor at right (text extends left), anchor at top (text extends down)
        xanchor = 'right'
        yanchor = 'top'
        x_offset = -3
        y_offset = -3
    elif 'xAI Colossus 1' in datacenter:
        # Bottom right: anchor at left (text extends right), anchor at top (text extends down)
        xanchor = 'left'
        yanchor = 'top'
        x_offset = 3
        y_offset = -3
    elif 'Google New Albany' in datacenter or 'ORNL Frontier' in datacenter:
        # Top left: anchor at right (text extends left), anchor at bottom (text extends up)
        xanchor = 'right'
        yanchor = 'bottom'
        x_offset = -3
        y_offset = 3
    else:
        # Upper right (default, including Colossus 2): anchor at left (text extends right), anchor at bottom (text extends up)
        xanchor = 'left'
        yanchor = 'bottom'
        x_offset = 3
        y_offset = 3

    # Add marker (all blue circles, using color from app.py)
    # X and Y are flipped: x is now workers, y is now MW/year
    fig.add_trace(go.Scatter(
        x=[workers],
        y=[mw_per_year],
        mode='markers',
        marker=dict(
            size=6,
            color='#5B8DBE',  # Blue color from app.py
            symbol='circle'
        ),
        hovertemplate='<b>' + datacenter + '</b><br>' +
                      'Workers: %{x:,.0f}<br>' +
                      'MW/year: %{y:.1f}<br>' +
                      '<extra></extra>',
        showlegend=False
    ))

    # Add text annotation with proper alignment
    fig.add_annotation(
        x=workers,
        y=mw_per_year,
        text=short_name,
        showarrow=False,
        xshift=x_offset,
        yshift=y_offset,
        font=dict(size=7),
        xanchor=xanchor,
        yanchor=yanchor
    )

# Legend removed per user request

# Calculate and add regression line (forcing through origin)
# X and Y are flipped: X is now workers, Y is now MW/year
if len(results_df) > 1:
    X = results_df['Workers'].values  # Now workers is X
    y = results_df['MW per Year'].values  # Now MW/year is Y

    # Fit linear regression through origin (intercept = 0)
    # slope = sum(x*y) / sum(x^2)
    slope = np.sum(X * y) / np.sum(X ** 2)
    intercept = 0

    # Calculate relative sigma for prediction intervals
    # This combines both uncertainty in the slope and inherent scatter

    # 1. Calculate residuals from the fitted line
    y_pred_points = slope * X
    residuals = y - y_pred_points

    # 2. Calculate relative residuals (residual / predicted value)
    relative_residuals = residuals / y_pred_points

    # 3. Standard deviation of relative residuals (inherent scatter)
    relative_sigma_residuals = np.std(relative_residuals, ddof=1)

    # 4. Standard error of the slope estimate
    # For regression through origin: SE(slope) = σ / sqrt(Σx²)
    # where σ is estimated from residuals
    residual_std = np.std(residuals, ddof=1)
    se_slope = residual_std / np.sqrt(np.sum(X ** 2))

    # 5. For a prediction at x_new, the relative uncertainty combines:
    #    - relative_sigma_residuals (inherent scatter, proportional to y)
    #    - uncertainty from slope: (se_slope * x_new) / (slope * x_new) = se_slope / slope
    relative_sigma_slope = se_slope / slope

    # 6. Combined relative sigma for prediction interval (add in quadrature)
    relative_sigma = np.sqrt(relative_sigma_residuals**2 + relative_sigma_slope**2)

    # Generate regression line from origin
    x_range = np.linspace(0, X.max(), 100)
    y_pred = slope * x_range

    # Calculate R² for regression through origin
    y_pred_points = slope * X
    ss_res = np.sum((y - y_pred_points) ** 2)
    ss_tot = np.sum(y ** 2)  # Total sum of squares (from origin)
    r2 = 1 - (ss_res / ss_tot)

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'y = {slope:.4f}x<br>σ/m = {relative_sigma:.2f}',
        showlegend=True
    ))

    print(f"\n\nRegression Results (through origin):")
    print(f"Slope (m): {slope:.6f} MW/year per worker")
    print(f"Standard error of slope: {se_slope:.6f}")
    print(f"Relative uncertainty in slope: {relative_sigma_slope:.4f} ({relative_sigma_slope*100:.2f}%)")
    print(f"\nRelative residual scatter: {relative_sigma_residuals:.4f} ({relative_sigma_residuals*100:.2f}%)")
    print(f"\nCombined Relative Sigma (for prediction intervals): {relative_sigma:.4f} ({relative_sigma*100:.2f}%)")
    print(f"  - This is sqrt(residual_scatter² + slope_uncertainty²)")
    print(f"\nSample size: {len(X)}")
    print(f"Intercept: {intercept:.6f} MW/year (forced to 0)")
    print(f"R²: {r2:.4f}")

# Update layout (6 inches x 3.5 inches at 100 DPI)
# X and Y axes are flipped
fig.update_layout(
    xaxis_title='Number of Concurrent Construction Workers',
    yaxis_title='Build Rate (MW per Year)',
    width=600,
    height=350,
    hovermode='closest',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    margin=dict(t=20, b=50, l=50, r=20)
)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Save to PNG (6 inches x 3.5 inches at 100 DPI)
fig.write_image('workers_vs_gw_per_year.png', width=600, height=350, scale=2)
print("\n\nPlot saved to workers_vs_gw_per_year.png")
