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
    const startYear = 2026;

    // Find the latest ASI milestone time across all trajectories
    let latestAsiTime = null;
    const trajectories = [mc.global, mc.covert, mc.prc_no_slowdown, mc.proxy_project];
    for (const traj of trajectories) {
        if (traj && traj.milestones_median && traj.milestones_median.ASI) {
            const asiTime = traj.milestones_median.ASI.time;
            if (asiTime && (latestAsiTime === null || asiTime > latestAsiTime)) {
                latestAsiTime = asiTime;
            }
        }
    }

    // Calculate end year: min of (latest ASI * 1.2) or (agreement_year + 10)
    const asiBasedEnd = latestAsiTime ? latestAsiTime * 1.2 : Infinity;
    const agreementBasedEnd = agreement_year ? agreement_year + 10 : Infinity;
    const endYear = Math.min(asiBasedEnd, agreementBasedEnd);

    const traces = [];

    // Color scheme: blue for US, purple/red for PRC
    const usColor = '#5B8DBE';    // Blue for US
    const prcColor = '#C77CAA';   // Purple/pink for PRC

    // Legend order: US no slowdown, PRC no slowdown, US slowdown, PRC slowdown
    // Add traces in this order for correct legend ordering

    // 1. US no slowdown (dashed) - Largest U.S. Company
    traces.push(...createMedianTrace(mc.global, usColor, 'US (no slowdown)', startYear, endYear, 'dash'));

    // Global milestone markers
    const milestoneDataGlobal = extractMilestoneData(mc.global.milestones_median, startYear, endYear);
    if (milestoneDataGlobal.length > 0) {
        traces.push({
            x: milestoneDataGlobal.map(m => m.time),
            y: milestoneDataGlobal.map(m => m.speedup),
            type: 'scatter',
            mode: 'markers+text',
            marker: {
                color: usColor,
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
                color: usColor,
                family: 'Arial, sans-serif'
            },
            hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
            showlegend: false
        });
    }

    // 2. PRC no slowdown (dashed)
    if (mc.prc_no_slowdown) {
        traces.push(...createMedianTrace(mc.prc_no_slowdown, prcColor, 'PRC (no slowdown)', startYear, endYear, 'dash'));

        // PRC no-slowdown milestone markers
        const milestoneDataNoSlowdown = extractMilestoneData(mc.prc_no_slowdown.milestones_median, startYear, endYear);
        if (milestoneDataNoSlowdown.length > 0) {
            traces.push({
                x: milestoneDataNoSlowdown.map(m => m.time),
                y: milestoneDataNoSlowdown.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: prcColor,
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
                    color: prcColor,
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // 3. US slowdown (solid) - US Frontier AI R&D
    if (mc.proxy_project) {
        traces.push(...createMedianTrace(mc.proxy_project, usColor, 'US (slowdown)', startYear, endYear));

        // US Frontier milestone markers
        const milestoneDataUSFrontier = extractMilestoneData(mc.proxy_project.milestones_median, startYear, endYear);
        if (milestoneDataUSFrontier.length > 0) {
            traces.push({
                x: milestoneDataUSFrontier.map(m => m.time),
                y: milestoneDataUSFrontier.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: usColor,
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
                    color: usColor,
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // 4. PRC slowdown (solid) - PRC Covert AI R&D
    if (mc.covert) {
        traces.push(...createMedianTrace(mc.covert, prcColor, 'PRC (slowdown)', startYear, endYear));

        // PRC covert milestone markers
        const milestoneDataCovert = extractMilestoneData(mc.covert.milestones_median, startYear, endYear);
        if (milestoneDataCovert.length > 0) {
            traces.push({
                x: milestoneDataCovert.map(m => m.time),
                y: milestoneDataCovert.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: prcColor,
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
                    color: prcColor,
                    family: 'Arial, sans-serif'
                },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // Add a dummy trace for the agreement line legend entry
    if (agreement_year) {
        traces.push({
            x: [null],
            y: [null],
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#888',
                width: 2,
                dash: 'dot'
            },
            name: 'Agreement start',
            showlegend: true
        });
    }

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            automargin: true,
            range: [startYear, endYear]
        },
        yaxis: {
            title: 'AI R&D Speedup',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            type: 'log',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 50, r: 20, t: 10, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        shapes: agreement_year ? [{
            type: 'line',
            x0: agreement_year,
            x1: agreement_year,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: '#888',
                width: 2,
                dash: 'dot'
            }
        }] : []
    };

    console.log('Calling Plotly.newPlot with', traces.length, 'traces');
    console.log('First trace x length:', traces[0]?.x?.length, 'y length:', traces[0]?.y?.length);
    console.log('First trace x sample:', traces[0]?.x?.slice(0, 3));
    console.log('First trace y sample:', traces[0]?.y?.slice(0, 3));
    // Clear any loading indicator before plotting
    const plotContainer = document.getElementById('slowdownPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '';
    }
    Plotly.newPlot('slowdownPlot', traces, layout, {displayModeBar: false, responsive: true});
    console.log('Plotly.newPlot completed');

    // Match plot heights to dashboard height after plot is created
    setTimeout(() => {
        const dashboard = document.querySelector('#agreementTopSection .dashboard');
        const plotContainers = document.querySelectorAll('#agreementTopSection .plot-container');
        if (dashboard && plotContainers.length > 0) {
            const dashboardHeight = dashboard.offsetHeight;
            plotContainers.forEach(container => {
                container.style.height = dashboardHeight + 'px';
                // Also set height on the plot div inside
                const plotDiv = container.querySelector('.plot');
                if (plotDiv) {
                    // Account for the title height and padding
                    const titleDiv = container.querySelector('.plot-title');
                    const titleHeight = titleDiv ? titleDiv.offsetHeight : 0;
                    const plotHeight = dashboardHeight - titleHeight - 50; // 50 for padding/margins
                    plotDiv.style.height = plotHeight + 'px';
                }
            });
            // Force resize after setting height
            setTimeout(() => {
                Plotly.Plots.resize('slowdownPlot');
                // Also resize other plots in the section if they exist
                if (document.getElementById('agreementRiskPlot')) {
                    Plotly.Plots.resize('agreementRiskPlot');
                }
                if (document.getElementById('agreementCapsPlot')) {
                    Plotly.Plots.resize('agreementCapsPlot');
                }
            }, 50);
        }
    }, 150);
}

// Show simple loading indicator (text only) and size container to match dashboard
function showTakeoffLoadingIndicator() {
    const plotContainer = document.getElementById('slowdownPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '<p style="text-align: center; color: #888; padding: 20px;">Loading...</p>';
    }

    // Size the container to match dashboard height immediately
    const dashboard = document.querySelector('#agreementTopSection .dashboard');
    const plotContainers = document.querySelectorAll('#agreementTopSection .plot-container');
    if (dashboard && plotContainers.length > 0) {
        const dashboardHeight = dashboard.offsetHeight;
        plotContainers.forEach(container => {
            container.style.height = dashboardHeight + 'px';
        });
    }
}

// Show loading indicator with progress bar (kept for future use)
function showTakeoffLoadingIndicatorWithProgress() {
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
        showTakeoffLoadingIndicatorWithProgress,
        updateTakeoffProgress,
        updateTakeoffStatus,
        showTakeoffError
    };
}
