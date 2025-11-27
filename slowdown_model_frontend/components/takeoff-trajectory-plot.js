// Takeoff Trajectory Plot Component JavaScript

// Helper function to create a median-only trace (no uncertainty bands)
function createMedianTrace(mcData, color, name, startYear, endYear, lineStyle = 'solid') {
    if (!mcData || !mcData.trajectory_times || !mcData.speedup_percentiles) {
        return [];
    }

    const times = mcData.trajectory_times;
    const median = mcData.speedup_percentiles.median;

    // Filter to year range
    const indices = times.map((t, i) => ({t, i})).filter(d => d.t >= startYear && d.t <= endYear);
    const filteredTimes = indices.map(d => times[d.i]);
    const filteredMedian = indices.map(d => median[d.i]);

    return [
        // Median line only
        {
            x: filteredTimes,
            y: filteredMedian,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: color,
                width: 3,
                dash: lineStyle === 'dash' ? 'dash' : undefined
            },
            name: name,
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        }
    ];
}

// Helper function to create a percentile band trace (filled area between p25 and p75)
// Used by the uncertainty plot
function createPercentileBand(mcData, color, name, startYear, endYear, lineStyle = 'solid') {
    if (!mcData || !mcData.trajectory_times || !mcData.speedup_percentiles) {
        return [];
    }

    const times = mcData.trajectory_times;
    const p25 = mcData.speedup_percentiles.p25;
    const p75 = mcData.speedup_percentiles.p75;
    const median = mcData.speedup_percentiles.median;

    // Filter to year range
    const indices = times.map((t, i) => ({t, i})).filter(d => d.t >= startYear && d.t <= endYear);
    const filteredTimes = indices.map(d => times[d.i]);
    const filteredP25 = indices.map(d => p25[d.i]);
    const filteredP75 = indices.map(d => p75[d.i]);
    const filteredMedian = indices.map(d => median[d.i]);

    // Parse color to get RGB values for transparency
    const hexToRgb = (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : {r: 0, g: 0, b: 0};
    };
    const rgb = hexToRgb(color);

    return [
        // Upper bound (p75) - invisible line
        {
            x: filteredTimes,
            y: filteredP75,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
        },
        // Filled area between p25 and p75
        {
            x: filteredTimes,
            y: filteredP25,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`,
            line: { color: 'transparent' },
            name: name + ' (25th-75th)',
            showlegend: false,
            hoverinfo: 'skip'
        },
        // Median line
        {
            x: filteredTimes,
            y: filteredMedian,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: color,
                width: 3,
                dash: lineStyle === 'dash' ? 'dash' : undefined
            },
            name: name,
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        }
    ];
}

// Define which milestones to display
const milestoneNames = {
    'AC': 'AC',
    'SAR-level-experiment-selection-skill': 'SAR',
    'SIAR-level-experiment-selection-skill': 'SIAR',
    'TED-AI': 'TED-AI',
    'ASI': 'ASI'
};

// Helper function to extract milestone data from milestones dict
function extractMilestoneData(milestones, startYear, endYear) {
    const milestoneData = [];
    if (!milestones) return milestoneData;
    for (const [key, label] of Object.entries(milestoneNames)) {
        if (milestones[key]) {
            const time = milestones[key].time;
            const progress_multiplier = milestones[key].progress_multiplier;

            if (progress_multiplier && !isNaN(progress_multiplier) && time >= startYear && time <= endYear) {
                milestoneData.push({
                    key: key,
                    label: label,
                    time: time,
                    speedup: progress_multiplier
                });
            }
        }
    }
    return milestoneData;
}

function plotTakeoffModel(data) {
    // Get Monte Carlo data
    const mc = data.monte_carlo || {};

    // Check for required Monte Carlo data
    if (!mc.global || !mc.global.trajectory_times || !mc.global.speedup_percentiles) {
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">No trajectory data available</p>';
        return;
    }

    // Get agreement start year from the data
    const agreement_year = data.agreement_year;

    // Filter trajectory data to start from 2026 (to show full trendline)
    // and end at the simulation end year (from covert compute data or MC trajectory times)
    const startYear = 2026;
    const combinedData = data.combined_covert_compute;
    const mcTrajectoryTimes = mc.global.trajectory_times;
    const simulationEndYear = combinedData && combinedData.years && combinedData.years.length > 0
        ? combinedData.years[combinedData.years.length - 1]
        : mcTrajectoryTimes[mcTrajectoryTimes.length - 1];
    const endYear = simulationEndYear;

    const traces = [];

    // Add Largest U.S. Company median line only (no uncertainty bands on main plot)
    traces.push(...createMedianTrace(mc.global, '#2A623D', 'Largest U.S. Company (no slowdown)', startYear, endYear));

    // Global milestone markers (from MC median run - aligned with the plotted line)
    const milestoneDataGlobal = extractMilestoneData(mc.global.milestones_median, startYear, endYear);
    if (milestoneDataGlobal.length > 0) {
        traces.push({
            x: milestoneDataGlobal.map(m => m.time),
            y: milestoneDataGlobal.map(m => m.speedup),
            type: 'scatter',
            mode: 'markers+text',
            marker: {
                color: '#2A623D',
                size: 10,
                line: {
                    color: 'white',
                    width: 2
                }
            },
            text: milestoneDataGlobal.map(m => m.label),
            textposition: 'top center',
            textfont: {
                size: 11,
                color: '#2A623D',
                family: 'Arial, sans-serif'
            },
            hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
            showlegend: false
        });
    }

    // Add PRC covert trajectory median line only
    if (mc.covert) {
        traces.push(...createMedianTrace(mc.covert, '#8B4513', 'PRC Covert AI R&D', startYear, endYear));

        // PRC covert milestone markers (from MC median run - aligned with the plotted line)
        const milestoneDataCovert = extractMilestoneData(mc.covert.milestones_median, startYear, endYear);
        if (milestoneDataCovert.length > 0) {
            traces.push({
                x: milestoneDataCovert.map(m => m.time),
                y: milestoneDataCovert.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: '#8B4513',
                    size: 10,
                    line: {
                        color: 'white',
                        width: 2
                    }
                },
                text: milestoneDataCovert.map(m => m.label),
                textposition: 'bottom center',
                textfont: {
                    size: 11,
                    color: '#8B4513',
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // Add PRC no-slowdown trajectory median line only
    if (mc.prc_no_slowdown) {
        traces.push(...createMedianTrace(mc.prc_no_slowdown, '#D2691E', 'PRC AI R&D (no slowdown)', startYear, endYear, 'dash'));

        // PRC no-slowdown milestone markers (from MC median run - aligned with the plotted line)
        const milestoneDataNoSlowdown = extractMilestoneData(mc.prc_no_slowdown.milestones_median, startYear, endYear);
        if (milestoneDataNoSlowdown.length > 0) {
            traces.push({
                x: milestoneDataNoSlowdown.map(m => m.time),
                y: milestoneDataNoSlowdown.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: '#D2691E',
                    size: 10,
                    symbol: 'diamond',
                    line: {
                        color: 'white',
                        width: 2
                    }
                },
                text: milestoneDataNoSlowdown.map(m => m.label),
                textposition: 'top center',
                textfont: {
                    size: 11,
                    color: '#D2691E',
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // Add US Frontier trajectory median line only
    if (mc.proxy_project) {
        traces.push(...createMedianTrace(mc.proxy_project, '#4169E1', 'US Frontier AI R&D', startYear, endYear));

        // US Frontier milestone markers (from MC median run - aligned with the plotted line)
        const milestoneDataUSFrontier = extractMilestoneData(mc.proxy_project.milestones_median, startYear, endYear);
        if (milestoneDataUSFrontier.length > 0) {
            traces.push({
                x: milestoneDataUSFrontier.map(m => m.time),
                y: milestoneDataUSFrontier.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: '#4169E1',
                    size: 10,
                    symbol: 'square',
                    line: {
                        color: 'white',
                        width: 2
                    }
                },
                text: milestoneDataUSFrontier.map(m => m.label),
                textposition: 'top center',
                textfont: {
                    size: 11,
                    color: '#4169E1',
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            automargin: true,
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        yaxis: {
            title: 'AI Software R&D Uplift (relative to human baseline)',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            automargin: true,
            type: 'log',
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#ddd',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 100, r: 40, t: 40, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa'
    };

    console.log('Calling Plotly.newPlot with', traces.length, 'traces');
    console.log('First trace x length:', traces[0]?.x?.length, 'y length:', traces[0]?.y?.length);
    console.log('First trace x sample:', traces[0]?.x?.slice(0, 3));
    console.log('First trace y sample:', traces[0]?.y?.slice(0, 3));
    Plotly.newPlot('slowdownPlot', traces, layout, {displayModeBar: false, responsive: true});
    console.log('Plotly.newPlot completed');
}

// Show loading indicator with progress bar
function showTakeoffLoadingIndicator() {
    const plotContainer = document.getElementById('slowdownPlot');
    if (plotContainer) {
        plotContainer.innerHTML = `
            <div style="text-align: center; padding: 100px 20px;">
                <p style="color: #888; margin-bottom: 15px;">Loading takeoff trajectories...</p>
                <div id="progressContainer" style="max-width: 400px; margin: 0 auto;">
                    <div style="background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                        <div id="progressBar" style="background: linear-gradient(90deg, #2A623D 0%, #4CAF50 100%); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
                    </div>
                    <p id="progressText" style="color: #666; font-size: 13px; margin-top: 8px;">Starting simulations...</p>
                </div>
            </div>
        `;
    }
}

// Update progress bar
function updateTakeoffProgress(current, total, trajectoryName) {
    const progressPercent = (current / total) * 100;
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    if (progressBar) {
        progressBar.style.width = progressPercent + '%';
    }
    if (progressText) {
        progressText.textContent = trajectoryName;
    }
}

// Update status text (for non-progress status messages)
function updateTakeoffStatus(message) {
    const progressText = document.getElementById('progressText');
    if (progressText) {
        progressText.textContent = message;
    }
}

// Show error message
function showTakeoffError(message) {
    const plotContainer = document.getElementById('slowdownPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">Error loading data: ' + message + '</p>';
    }
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createPercentileBand,
        milestoneNames,
        extractMilestoneData,
        plotTakeoffModel,
        showTakeoffLoadingIndicator,
        updateTakeoffProgress,
        updateTakeoffStatus,
        showTakeoffError
    };
}
