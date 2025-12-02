// Agreement Section Component JavaScript

// Helper function to create a median-only trace (no uncertainty bands)
function createMedianTrace(mcData, color, name, startYear, endYear, lineStyle = 'solid', opacity = 1.0) {
    if (!mcData || !mcData.trajectory_times || !mcData.speedup_percentiles) {
        return [];
    }

    const times = mcData.trajectory_times;
    const median = mcData.speedup_percentiles.median;

    // Filter to year range
    const indices = times.map((t, i) => ({t, i})).filter(d => d.t >= startYear && d.t <= endYear);
    const filteredTimes = indices.map(d => times[d.i]);
    const filteredMedian = indices.map(d => median[d.i]);

    // Determine dash style
    let dashStyle = undefined;
    if (lineStyle === 'dash') dashStyle = 'dash';
    else if (lineStyle === 'dot') dashStyle = 'dot';

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
                dash: dashStyle
            },
            opacity: opacity,
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

// Plot the takeoff model trajectories
function plotTakeoffModel(data) {
    // Get Monte Carlo data
    const mc = data.monte_carlo || {};

    // Check for required Monte Carlo data
    if (!mc.global || !mc.global.trajectory_times || !mc.global.speedup_percentiles) {
        const plotContainer = document.getElementById('slowdownPlot');
        if (plotContainer) {
            plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">No trajectory data available</p>';
        }
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
                line: { color: 'white', width: 2 }
            },
            text: milestoneDataGlobal.map(m => m.label),
            textposition: 'top center',
            textfont: { size: 11, color: usColor, family: 'Arial, sans-serif' },
            hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
            showlegend: false
        });
    }

    // 2. PRC no slowdown (dashed)
    if (mc.prc_no_slowdown) {
        traces.push(...createMedianTrace(mc.prc_no_slowdown, prcColor, 'PRC (no slowdown)', startYear, endYear, 'dash'));
    }

    // 3. Proxy Project (dotted black)
    if (mc.proxy_project) {
        traces.push(...createMedianTrace(mc.proxy_project, '#000000', 'Proxy Project', startYear, endYear, 'dot'));
    }

    // 4. Capability Cap (solid black with opacity)
    if (mc.capability_cap) {
        traces.push(...createMedianTrace(mc.capability_cap, '#000000', 'Capability Cap', startYear, endYear, 'solid', 0.5));
    }

    // 5. PRC slowdown (solid) - PRC Covert AI R&D
    if (mc.covert) {
        traces.push(...createMedianTrace(mc.covert, prcColor, 'PRC (slowdown)', startYear, endYear));

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
                    line: { color: 'white', width: 2 }
                },
                text: milestoneDataCovert.map(m => m.label),
                textposition: 'bottom center',
                textfont: { size: 11, color: prcColor, family: 'Arial, sans-serif' },
                hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
                showlegend: false
            });
        }
    }

    // 6. US slowdown (solid blue)
    if (mc.us_slowdown) {
        traces.push(...createMedianTrace(mc.us_slowdown, usColor, 'US (slowdown)', startYear, endYear));

        const milestoneDataUSSlowdown = extractMilestoneData(mc.us_slowdown.milestones_median, startYear, endYear);
        if (milestoneDataUSSlowdown.length > 0) {
            traces.push({
                x: milestoneDataUSSlowdown.map(m => m.time),
                y: milestoneDataUSSlowdown.map(m => m.speedup),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    color: usColor,
                    size: 10,
                    line: { color: 'white', width: 2 }
                },
                text: milestoneDataUSSlowdown.map(m => m.label),
                textposition: 'bottom center',
                textfont: { size: 11, color: usColor, family: 'Arial, sans-serif' },
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
            line: { color: '#888', width: 2, dash: 'dot' },
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
            line: { color: '#888', width: 2, dash: 'dot' }
        }] : []
    };

    // Clear any loading indicator before plotting
    const plotContainer = document.getElementById('slowdownPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '';
    }
    Plotly.newPlot('slowdownPlot', traces, layout, {displayModeBar: false, responsive: true});

    // Match plot heights to dashboard height after plot is created
    setTimeout(() => {
        const dashboard = document.querySelector('#agreementTopSection .dashboard');
        const plotContainers = document.querySelectorAll('#agreementTopSection .plot-container');
        if (dashboard && plotContainers.length > 0) {
            const dashboardHeight = dashboard.offsetHeight;
            plotContainers.forEach(container => {
                container.style.height = dashboardHeight + 'px';
                const plotDiv = container.querySelector('.plot');
                if (plotDiv) {
                    const titleDiv = container.querySelector('.plot-title');
                    const titleHeight = titleDiv ? titleDiv.offsetHeight : 0;
                    const plotHeight = dashboardHeight - titleHeight - 50;
                    plotDiv.style.height = plotHeight + 'px';
                }
            });
            setTimeout(() => {
                Plotly.Plots.resize('slowdownPlot');
                if (document.getElementById('agreementComputePlot')) {
                    Plotly.Plots.resize('agreementComputePlot');
                }
            }, 50);
        }
    }, 150);
}

// Load the agreement section HTML
async function loadAgreementSection() {
    const container = document.getElementById('agreement-section-container');
    if (!container) return;

    try {
        const response = await fetch('/slowdown_model_frontend/components/agreement-section.html');
        const html = await response.text();
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading agreement section:', error);
    }
}

// Plot compute over time in agreement section (styled to match takeoff trajectory plot)
function plotAgreementComputeOverTime(data) {
    const combinedData = data.combined_covert_compute;
    const largestCompanyData = data.largest_company_compute;
    const prcNoSlowdownData = data.prc_no_slowdown_compute;
    const proxyProjectData = data.proxy_project_compute;
    const usSlowdownData = data.us_slowdown_compute;  // Effective compute for US slowdown

    const container = document.getElementById('agreementComputePlot');
    if (!container) return;

    if (!combinedData || !combinedData.years || !combinedData.median) {
        container.innerHTML = '<p style="text-align: center; color: #888; padding: 20px;">No compute data available</p>';
        return;
    }

    const years = combinedData.years;
    const median = combinedData.median;
    const agreement_year = data.agreement_year;

    // Color scheme matching takeoff trajectory plot
    const usColor = '#5B8DBE';    // Blue for US
    const prcColor = '#C77CAA';   // Purple/pink for PRC

    const traces = [];

    // 1. US no slowdown (dashed) - Largest U.S. Company
    if (largestCompanyData && largestCompanyData.years && largestCompanyData.compute) {
        traces.push({
            x: largestCompanyData.years,
            y: largestCompanyData.compute,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: usColor,
                width: 3,
                dash: 'dash'
            },
            name: 'US (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 1.5. US with slowdown (solid) - Effective compute during AI slowdown
    if (usSlowdownData && usSlowdownData.years && usSlowdownData.compute) {
        // Plot the full trajectory (no filtering)
        traces.push({
            x: usSlowdownData.years,
            y: usSlowdownData.compute,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: usColor,
                width: 3
            },
            name: 'US (with slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 2. PRC no slowdown (dashed)
    if (prcNoSlowdownData && prcNoSlowdownData.years && prcNoSlowdownData.median) {
        traces.push({
            x: prcNoSlowdownData.years,
            y: prcNoSlowdownData.median,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: prcColor,
                width: 3,
                dash: 'dash'
            },
            name: 'PRC (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 3. Proxy Project (dotted black) - only from agreement year onwards
    if (proxyProjectData && proxyProjectData.years && proxyProjectData.compute) {
        // Filter to only include years from agreement year onwards
        const proxyYears = [];
        const proxyCompute = [];
        for (let i = 0; i < proxyProjectData.years.length; i++) {
            if (proxyProjectData.years[i] >= agreement_year) {
                proxyYears.push(proxyProjectData.years[i]);
                proxyCompute.push(proxyProjectData.compute[i]);
            }
        }

        if (proxyYears.length > 0) {
            traces.push({
                x: proxyYears,
                y: proxyCompute,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#000000',
                    width: 3,
                    dash: 'dot'
                },
                name: 'Proxy Project',
                hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
            });
        }
    }

    // 4. PRC covert compute (solid)
    traces.push({
        x: years,
        y: median,
        type: 'scatter',
        mode: 'lines',
        line: {
            color: prcColor,
            width: 3
        },
        name: 'PRC (with slowdown)',
        hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
    });

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

    // Determine end year from the data
    const endYear = years[years.length - 1];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            automargin: true,
            range: [2026, endYear]
        },
        yaxis: {
            title: 'H100-equivalents',
            titlefont: { size: 11 },
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

    // Clear container before plotting
    container.innerHTML = '';

    Plotly.newPlot('agreementComputePlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Update the agreement section with data
function updateAgreementSection(data) {
    // Update speedup value in title
    const speedupValue = document.getElementById('agreement-speedup-value');
    if (speedupValue && data.speedup) {
        speedupValue.textContent = `${data.speedup}x AI R&D`;
    }

    // Update outcome values
    if (data.outcomes) {
        const slowdownEl = document.getElementById('outcome-slowdown');
        if (slowdownEl && data.outcomes.slowdown !== undefined) {
            slowdownEl.textContent = data.outcomes.slowdown;
        }

        const takeoverEl = document.getElementById('outcome-takeover-risk');
        if (takeoverEl && data.outcomes.takeoverRisk !== undefined) {
            takeoverEl.innerHTML = data.outcomes.takeoverRisk;
        }

        const prcEl = document.getElementById('outcome-prc-risk');
        if (prcEl && data.outcomes.prcRisk !== undefined) {
            prcEl.innerHTML = data.outcomes.prcRisk;
        }

        const computeReductionEl = document.getElementById('outcome-compute-reduction');
        if (computeReductionEl && data.outcomes.computeReduction !== undefined) {
            computeReductionEl.textContent = data.outcomes.computeReduction;
        }

        const researchReductionEl = document.getElementById('outcome-research-reduction');
        if (researchReductionEl && data.outcomes.researchReduction !== undefined) {
            researchReductionEl.textContent = data.outcomes.researchReduction;
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', loadAgreementSection);

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { loadAgreementSection, updateAgreementSection, plotAgreementComputeOverTime };
}
