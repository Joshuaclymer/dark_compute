// Takeoff Trajectory Plot Component JavaScript

// Helper function to create a percentile band trace (filled area between p25 and p75)
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
    if (!data.takeoff_trajectories) {
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">No trajectory data available</p>';
        return;
    }

    const trajectories = data.takeoff_trajectories;
    const milestones_global = trajectories.milestones_global;
    const trajectory_times = trajectories.trajectory_times;
    const global_ai_speedup = trajectories.global_ai_speedup;
    const covert_ai_speedup = trajectories.covert_ai_speedup;
    const covert_trajectory_times = trajectories.covert_trajectory_times;
    const milestones_covert = trajectories.milestones_covert;

    // Get Monte Carlo data
    const mc = data.monte_carlo || {};

    if (!milestones_global || !trajectory_times || !global_ai_speedup) {
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">Trajectory prediction failed - missing AI speedup data</p>';
        return;
    }

    // Get agreement start year from the data
    const agreement_year = data.agreement_year;

    // Filter trajectory data to start from 2026 (to show full trendline)
    // and end at the simulation end year (from covert compute data)
    const startYear = 2026;
    const combinedData = data.combined_covert_compute;
    const simulationEndYear = combinedData && combinedData.years && combinedData.years.length > 0
        ? combinedData.years[combinedData.years.length - 1]
        : trajectory_times[trajectory_times.length - 1];
    const endYear = simulationEndYear;

    const traces = [];

    // Add Global AI R&D with Monte Carlo percentile bands
    if (mc.global) {
        traces.push(...createPercentileBand(mc.global, '#2A623D', 'Global AI R&D (no slowdown)', startYear, endYear));

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
    } else {
        // Fallback to legacy single-trajectory mode
        const filteredData = trajectory_times.map((time, i) => ({
            time,
            speedup: global_ai_speedup[i]
        })).filter(d => d.time >= startYear && d.time <= endYear);

        traces.push({
            x: filteredData.map(d => d.time),
            y: filteredData.map(d => d.speedup),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#2A623D',
                width: 3
            },
            name: 'Global AI R&D (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        });

        // Global milestone markers (using legacy trajectory milestones)
        const milestoneDataGlobal = extractMilestoneData(milestones_global, startYear, endYear);
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
    }

    // Add PRC covert trajectory with Monte Carlo percentile bands
    if (mc.covert) {
        traces.push(...createPercentileBand(mc.covert, '#8B4513', 'PRC Covert AI R&D', startYear, endYear));

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
    } else if (covert_ai_speedup && covert_ai_speedup.length > 0 && covert_trajectory_times && covert_trajectory_times.length > 0) {
        // Fallback to legacy single-trajectory mode
        const filteredCovertData = covert_trajectory_times.map((time, i) => ({
            time,
            speedup: covert_ai_speedup[i]
        })).filter(d => d.time >= startYear && d.time <= endYear);

        // PRC covert trajectory line
        traces.push({
            x: filteredCovertData.map(d => d.time),
            y: filteredCovertData.map(d => d.speedup),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#8B4513',
                width: 3
            },
            name: 'PRC Covert AI R&D',
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        });

        // PRC covert milestone markers (using legacy trajectory milestones)
        const milestoneDataCovert = extractMilestoneData(milestones_covert, startYear, endYear);
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

    // Add PRC no-slowdown trajectory with Monte Carlo percentile bands
    if (mc.prc_no_slowdown) {
        traces.push(...createPercentileBand(mc.prc_no_slowdown, '#D2691E', 'PRC AI R&D (no slowdown)', startYear, endYear, 'dash'));

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
    } else {
        // Fallback to legacy single-trajectory mode
        const prcNoSlowdownTrajectories = data.prc_no_slowdown_trajectories;
        if (prcNoSlowdownTrajectories && prcNoSlowdownTrajectories.covert_ai_speedup && prcNoSlowdownTrajectories.covert_trajectory_times) {
            const noSlowdownSpeedup = prcNoSlowdownTrajectories.covert_ai_speedup;
            const noSlowdownTimes = prcNoSlowdownTrajectories.covert_trajectory_times;

            const filteredNoSlowdownData = noSlowdownTimes.map((time, i) => ({
                time,
                speedup: noSlowdownSpeedup[i]
            })).filter(d => d.time >= startYear && d.time <= endYear);

            // PRC no-slowdown trajectory line (dashed)
            traces.push({
                x: filteredNoSlowdownData.map(d => d.time),
                y: filteredNoSlowdownData.map(d => d.speedup),
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#D2691E',
                    width: 3,
                    dash: 'dash'
                },
                name: 'PRC AI R&D (no slowdown)',
                hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
            });

        // Extract no-slowdown milestone data for markers
        const milestones_no_slowdown = prcNoSlowdownTrajectories.milestones_covert;
        const milestoneDataNoSlowdown = [];
        if (milestones_no_slowdown) {
            for (const [key, label] of Object.entries(milestoneNames)) {
                if (milestones_no_slowdown[key]) {
                    const time = milestones_no_slowdown[key].time;
                    const progress_multiplier = milestones_no_slowdown[key].progress_multiplier;

                    if (progress_multiplier && !isNaN(progress_multiplier) && time >= startYear && time <= endYear) {
                        milestoneDataNoSlowdown.push({
                            key: key,
                            label: label,
                            time: time,
                            speedup: progress_multiplier
                        });
                    }
                }
            }
        }

        // PRC no-slowdown milestone markers
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
    }

    // Add Proxy Project trajectory with Monte Carlo percentile bands
    if (mc.proxy_project) {
        traces.push(...createPercentileBand(mc.proxy_project, '#4169E1', 'Proxy Project AI R&D', startYear, endYear));
    } else {
        // Fallback to legacy single-trajectory mode
        const proxyProjectTrajectories = data.proxy_project_trajectories;
    if (proxyProjectTrajectories && proxyProjectTrajectories.covert_ai_speedup && proxyProjectTrajectories.covert_trajectory_times) {
        const proxySpeedup = proxyProjectTrajectories.covert_ai_speedup;
        const proxyTimes = proxyProjectTrajectories.covert_trajectory_times;

        const filteredProxyData = proxyTimes.map((time, i) => ({
            time,
            speedup: proxySpeedup[i]
        })).filter(d => d.time >= startYear && d.time <= endYear);

        // Proxy Project trajectory line (step-change pattern)
        traces.push({
            x: filteredProxyData.map(d => d.time),
            y: filteredProxyData.map(d => d.speedup),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#4169E1',
                width: 3
            },
            name: 'Proxy Project AI R&D',
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        });

        // Extract proxy project milestone data for markers
        const milestones_proxy = proxyProjectTrajectories.milestones_covert;
        const milestoneDataProxy = [];
        if (milestones_proxy) {
            for (const [key, label] of Object.entries(milestoneNames)) {
                if (milestones_proxy[key]) {
                    const time = milestones_proxy[key].time;
                    const progress_multiplier = milestones_proxy[key].progress_multiplier;

                    if (progress_multiplier && !isNaN(progress_multiplier) && time >= startYear && time <= endYear) {
                        milestoneDataProxy.push({
                            key: key,
                            label: label,
                            time: time,
                            speedup: progress_multiplier
                        });
                    }
                }
            }
        }

        // Proxy Project milestone markers
        if (milestoneDataProxy.length > 0) {
            traces.push({
                x: milestoneDataProxy.map(m => m.time),
                y: milestoneDataProxy.map(m => m.speedup),
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
                text: milestoneDataProxy.map(m => m.label),
                textposition: 'bottom center',
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

    Plotly.newPlot('slowdownPlot', traces, layout, {displayModeBar: false, responsive: true});
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
        progressText.textContent = `${trajectoryName}: ${current} of ${total} simulations`;
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
        showTakeoffError
    };
}
