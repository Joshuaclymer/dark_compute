// Covert Uncertainty Plot Component JavaScript

// Show loading indicator for covert uncertainty plot
function showCovertUncertaintyLoadingIndicator() {
    const plotContainer = document.getElementById('covertUncertaintyPlot');
    if (plotContainer) {
        plotContainer.innerHTML = `
            <div style="text-align: center; padding: 100px 20px;">
                <p style="color: #888; margin-bottom: 15px;">Loading uncertainty data...</p>
                <div style="max-width: 400px; margin: 0 auto;">
                    <div style="background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                        <div id="covertUncertaintyProgressBar" style="background: linear-gradient(90deg, #8B4513 0%, #D2691E 100%); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
                    </div>
                    <p id="covertUncertaintyProgressText" style="color: #666; font-size: 13px; margin-top: 8px;">Waiting for simulations...</p>
                </div>
            </div>
        `;
    }
}

// Update progress bar for covert uncertainty plot
function updateCovertUncertaintyProgress(current, total, trajectoryName) {
    const progressPercent = (current / total) * 100;
    const progressBar = document.getElementById('covertUncertaintyProgressBar');
    const progressText = document.getElementById('covertUncertaintyProgressText');
    if (progressBar) {
        progressBar.style.width = progressPercent + '%';
    }
    if (progressText) {
        progressText.textContent = `${trajectoryName} (${current}/${total} - ${Math.round(progressPercent)}%)`;
    }
}

// Show error message for covert uncertainty plot
function showCovertUncertaintyError(message) {
    const plotContainer = document.getElementById('covertUncertaintyPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">Error loading data: ' + message + '</p>';
    }
}

function plotCovertUncertainty(data) {
    const plotContainer = document.getElementById('covertUncertaintyPlot');
    if (!plotContainer) {
        console.error('covertUncertaintyPlot container not found');
        return;
    }

    // Handle null or missing data
    if (!data) {
        plotContainer.innerHTML =
            '<p style="text-align: center; color: #888; padding: 50px;">No data available for uncertainty plot</p>';
        return;
    }

    // Get Monte Carlo data
    const mc = data.monte_carlo || {};

    // Check for required data - need at least covert MC data
    if (!mc.covert || !mc.covert.trajectory_times || !mc.covert.speedup_percentiles) {
        plotContainer.innerHTML =
            '<p style="text-align: center; color: #888; padding: 50px;">No covert compute uncertainty data available</p>';
        return;
    }

    // Get agreement start year from the data
    const agreement_year = data.agreement_year;

    // Filter trajectory data
    const startYear = 2026;
    const combinedData = data.combined_covert_compute;
    const mcTrajectoryTimes = mc.covert.trajectory_times;
    const simulationEndYear = combinedData && combinedData.years && combinedData.years.length > 0
        ? combinedData.years[combinedData.years.length - 1]
        : mcTrajectoryTimes[mcTrajectoryTimes.length - 1];
    const endYear = simulationEndYear;

    const traces = [];

    // Check if createPercentileBand is available
    if (typeof createPercentileBand !== 'function') {
        console.error('createPercentileBand function not found!');
        plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">Error: createPercentileBand function not available</p>';
        return;
    }

    // Helper function to filter trajectory data to year range
    const filterToYearRange = (times, values, startYear, endYear) => {
        const filteredTimes = [];
        const filteredValues = [];
        for (let i = 0; i < times.length; i++) {
            if (times[i] >= startYear && times[i] <= endYear) {
                filteredTimes.push(times[i]);
                filteredValues.push(values[i]);
            }
        }
        return { times: filteredTimes, values: filteredValues };
    };

    // Add all individual MC runs as very light traces (plot these first so they're behind)
    // PRC Covert MC runs
    if (mc.covert && mc.covert.all_mc_runs && mc.covert.all_mc_runs.length > 0) {
        const covertTimes = mc.covert.trajectory_times;
        mc.covert.all_mc_runs.forEach((runSpeedup, idx) => {
            const filtered = filterToYearRange(covertTimes, runSpeedup, startYear, endYear);
            traces.push({
                x: filtered.times,
                y: filtered.values,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'rgba(139, 69, 19, 0.15)',  // Very light brown
                    width: 1
                },
                showlegend: false,
                hoverinfo: 'skip'
            });
        });
    }

    // US Frontier MC runs
    if (mc.proxy_project && mc.proxy_project.all_mc_runs && mc.proxy_project.all_mc_runs.length > 0) {
        const proxyTimes = mc.proxy_project.trajectory_times;
        mc.proxy_project.all_mc_runs.forEach((runSpeedup, idx) => {
            const filtered = filterToYearRange(proxyTimes, runSpeedup, startYear, endYear);
            traces.push({
                x: filtered.times,
                y: filtered.values,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'rgba(156, 39, 176, 0.15)',  // Very light purple
                    width: 1
                },
                showlegend: false,
                hoverinfo: 'skip'
            });
        });
    }

    // Add PRC Covert AI R&D with uncertainty band (on top of light traces)
    const covertTraces = createPercentileBand(mc.covert, '#8B4513', 'PRC Covert AI R&D', startYear, endYear);
    traces.push(...covertTraces);

    // Add US Frontier with uncertainty band (on top of light traces)
    if (mc.proxy_project) {
        const proxyTraces = createPercentileBand(mc.proxy_project, '#9C27B0', 'US Frontier', startYear, endYear);
        traces.push(...proxyTraces);
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
        plot_bgcolor: '#fafafa',
        annotations: [{
            x: 0.5,
            y: -0.15,
            xref: 'paper',
            yref: 'paper',
            text: 'Light traces show individual Monte Carlo simulations; shaded regions show 25th-75th percentile',
            showarrow: false,
            font: { size: 11, color: '#666' }
        }]
    };

    try {
        // Clear loading indicator before plotting
        plotContainer.innerHTML = '';
        Plotly.newPlot('covertUncertaintyPlot', traces, layout, {displayModeBar: false, responsive: true});
    } catch (error) {
        console.error('Error creating covert uncertainty plot:', error);
        plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">Error rendering plot: ' + error.message + '</p>';
    }
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotCovertUncertainty };
}
