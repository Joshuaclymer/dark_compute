import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Find all shapefiles in China directory
china_dir = Path("extracted_data/China")
shapefiles = list(china_dir.rglob("*.shp"))
print(f"Found {len(shapefiles)} shapefiles in China directory")

# Sample parameters
SAMPLE_RATE = 0.01  # Sample 1% of buildings from each file
areas = []
total_buildings_count = 0

print("\nProcessing shapefiles with random sampling...")
for shp_file in tqdm(shapefiles):
    try:
        # Read shapefile
        gdf = gpd.read_file(shp_file)

        # Count total buildings
        num_buildings = len(gdf)
        total_buildings_count += num_buildings

        # Random sample
        sample_size = max(1, int(num_buildings * SAMPLE_RATE))
        if num_buildings > sample_size:
            gdf_sample = gdf.sample(n=sample_size, random_state=42)
        else:
            gdf_sample = gdf

        # Calculate areas in square meters
        # Reproject to a metric CRS for accurate area calculation (China Albers Equal Area)
        gdf_sample = gdf_sample.to_crs("EPSG:3395")  # World Mercator (good approximation for China)
        gdf_sample['area_m2'] = gdf_sample.geometry.area

        # Add to our list
        areas.extend(gdf_sample['area_m2'].tolist())

    except Exception as e:
        print(f"Error processing {shp_file}: {e}")
        continue

print(f"\nTotal buildings in China: {total_buildings_count:,}")
print(f"Sampled buildings: {len(areas):,}")
print(f"Effective sampling rate: {len(areas)/total_buildings_count*100:.2f}%")

# Convert to numpy array for analysis
areas = np.array(areas)

# Remove outliers (buildings > 100,000 m² are likely data errors or very special cases)
areas_filtered = areas[areas <= 100000]
print(f"\nBuildings after filtering (area <= 100,000 m²): {len(areas_filtered):,}")
print(f"Removed {len(areas) - len(areas_filtered):,} outliers")

# Statistics
print(f"\nArea Statistics (m²):")
print(f"  Mean: {np.mean(areas_filtered):.2f}")
print(f"  Median: {np.median(areas_filtered):.2f}")
print(f"  Std Dev: {np.std(areas_filtered):.2f}")
print(f"  Min: {np.min(areas_filtered):.2f}")
print(f"  Max: {np.max(areas_filtered):.2f}")
print(f"  25th percentile: {np.percentile(areas_filtered, 25):.2f}")
print(f"  75th percentile: {np.percentile(areas_filtered, 75):.2f}")
print(f"  95th percentile: {np.percentile(areas_filtered, 95):.2f}")
print(f"  99th percentile: {np.percentile(areas_filtered, 99):.2f}")

# Create histogram using Plotly
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=areas_filtered,
    nbinsx=100,
    name='Building Footprint Areas',
    marker_color='steelblue',
    opacity=0.75
))

fig.update_layout(
    title=f'Distribution of Building Footprint Areas in China<br><sub>Sample of {len(areas_filtered):,} buildings from {total_buildings_count:,} total buildings</sub>',
    xaxis_title='Building Footprint Area (m²)',
    yaxis_title='Frequency',
    showlegend=False,
    template='plotly_white',
    hovermode='x'
)

# Save as HTML
fig.write_html("china_building_areas_histogram.html")
print("\nHistogram saved as 'china_building_areas_histogram.html'")

# Create a log-scale version for better visualization
fig_log = go.Figure()

fig_log.add_trace(go.Histogram(
    x=areas_filtered,
    nbinsx=100,
    name='Building Footprint Areas',
    marker_color='steelblue',
    opacity=0.75
))

fig_log.update_layout(
    title=f'Distribution of Building Footprint Areas in China (Log Scale)<br><sub>Sample of {len(areas_filtered):,} buildings from {total_buildings_count:,} total buildings</sub>',
    xaxis_title='Building Footprint Area (m²)',
    yaxis_title='Frequency (log scale)',
    yaxis_type='log',
    showlegend=False,
    template='plotly_white',
    hovermode='x'
)

fig_log.write_html("china_building_areas_histogram_log.html")
print("Log-scale histogram saved as 'china_building_areas_histogram_log.html'")

# Save summary statistics to a file
with open('china_building_summary.txt', 'w') as f:
    f.write(f"China Building Footprint Analysis\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Total buildings in China: {total_buildings_count:,}\n")
    f.write(f"Sampled buildings: {len(areas):,}\n")
    f.write(f"Sampling rate: {len(areas)/total_buildings_count*100:.2f}%\n\n")
    f.write(f"Area Statistics (m²) [after filtering]:\n")
    f.write(f"  Mean: {np.mean(areas_filtered):.2f}\n")
    f.write(f"  Median: {np.median(areas_filtered):.2f}\n")
    f.write(f"  Std Dev: {np.std(areas_filtered):.2f}\n")
    f.write(f"  Min: {np.min(areas_filtered):.2f}\n")
    f.write(f"  Max: {np.max(areas_filtered):.2f}\n")
    f.write(f"  25th percentile: {np.percentile(areas_filtered, 25):.2f}\n")
    f.write(f"  75th percentile: {np.percentile(areas_filtered, 75):.2f}\n")
    f.write(f"  95th percentile: {np.percentile(areas_filtered, 95):.2f}\n")
    f.write(f"  99th percentile: {np.percentile(areas_filtered, 99):.2f}\n")

print("\nSummary saved as 'china_building_summary.txt'")
