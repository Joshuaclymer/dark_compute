"""
Generate H100 equivalents per wafer vs process node plot using Plotly with consistent styling.
Shows how compute capacity per wafer changes with process node.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '../..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# Node data
h100_node = 4
TPP_per_transistor_density = 644

process_node = [3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 130, 180]

transistor_density = [215.6, 145.86, 137.6, 106.96, 90.64, 60.3, 33.8, 28.88, 16.50, 14.44, 7.22, 3.61, 1.80, 0.90, 0.45]

year_process_node_first_reached_high_volume_manufacturing = [
    2023, 2022, 2020, 2020, 2018, 2017, 2018, 2015, 2012, 2011, 2009, 2006, 2004, 2001, 1999
]

# H100 specs
h100_tpp = 63328

# Calculate TPP for each node based on transistor density
tpp = [td * TPP_per_transistor_density for td in transistor_density]

# Calculate H100 equivalents per wafer
dies_per_wafer = 28
h100_equiv_per_wafer = [(dies_per_wafer * tpp_val) / h100_tpp for tpp_val in tpp]

# Find H100 index
h100_index = process_node.index(4)

# Regression analysis
process_node_array = np.array(process_node)
density_array = np.array(transistor_density)
log_process_node = np.log(process_node_array)
log_density = np.log(density_array)
coefficients = np.polyfit(log_process_node, log_density, 1)
a, b = coefficients

# Calculate R-squared
log_density_fit = a * log_process_node + b
residuals = log_density - log_density_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_density - np.mean(log_density))**2)
r_squared = 1 - (ss_res / ss_tot)

# Create figure
fig = go.Figure()

# Add line connecting all data points
fig.add_trace(go.Scatter(
    x=process_node, y=h100_equiv_per_wafer,
    mode='lines',
    line=dict(color=STYLE['blue'], width=STYLE['line_width']),
    name='H100 equivalents',
    hoverinfo='skip'
))

# Add scatter points
hover_text = [f'Node: {n}nm<br>Year: {y}<br>H100 equiv: {h:.2f}'
              for n, y, h in zip(process_node, year_process_node_first_reached_high_volume_manufacturing, h100_equiv_per_wafer)]
fig.add_trace(go.Scatter(
    x=process_node, y=h100_equiv_per_wafer,
    mode='markers',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue'], line=dict(color='black', width=STYLE['marker_line_width'])),
    name='Process nodes',
    text=hover_text,
    hovertemplate='%{text}<extra></extra>'
))


# Add regression line
process_node_fit = np.logspace(np.log10(min(process_node)), np.log10(max(process_node)), 100)
density_fit = np.exp(b) * process_node_fit**a
h100_equiv_fit = [(dies_per_wafer * d * TPP_per_transistor_density) / h100_tpp for d in density_fit]
fig.add_trace(go.Scatter(
    x=process_node_fit, y=h100_equiv_fit,
    mode='lines',
    line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash'),
    name=f'Power law fit (R2={r_squared:.3f})'
))

apply_common_layout(
    fig,
    xaxis_title='TSMC Process Node (nm)',
    yaxis_title='H100 Equivalents per Wafer',
    xaxis_log=True,
    yaxis_log=True,
    legend_position='bottom_left',
    show_legend=True
)

fig.update_xaxes(
    autorange='reversed',
    tickvals=process_node,
    ticktext=[str(n) for n in process_node],
    tickfont=dict(size=9)
)
fig.update_yaxes(range=[np.log10(0.01), np.log10(100)])

save_plot(fig, 'h100_equiv_per_year_vs_process_node.png')
save_html(fig, 'h100_equiv_per_year_vs_process_node.html')

print(f"Power law fit: density = {np.exp(b):.4f} * node^{a:.4f}")
print(f"R2 = {r_squared:.4f}")
