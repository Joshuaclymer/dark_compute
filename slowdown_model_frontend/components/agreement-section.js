// Agreement Section Component JavaScript

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

// Show loading indicator for agreement risk plot and size container to match dashboard
function showAgreementRiskLoadingIndicator() {
    const plotContainer = document.getElementById('agreementRiskPlot');
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

// Plot risk reduction vs slowdown duration
function plotAgreementRiskOverTime(data) {
    const plotData = data.risk_reduction_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('agreementRiskPlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 20px;">No data available</p>';
        }
        return;
    }

    // Colors consistent with dark compute main page
    const aiTakeoverColor = '#E8A863';      // Orange for AI Takeover (matches AI R&D color)
    const humanPowerGrabsColor = '#5AA89B'; // Turquoise green for Human Power Grabs
    const catastropheColor = '#5B8DBE';     // Blue for combined P(Domestic Takeover) (matches probability color)

    const traces = [
        // Combined risk reduction line first (main focus, thicker)
        {
            x: plotData.slowdown_duration,
            y: plotData.risk_reduction,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: catastropheColor,
                width: 3
            },
            name: 'Domestic Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        },
        // AI Takeover risk reduction line
        {
            x: plotData.slowdown_duration,
            y: plotData.p_ai_takeover_reduction,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: aiTakeoverColor,
                width: 2
            },
            name: 'AI Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        },
        // Human Power Grabs risk reduction line
        {
            x: plotData.slowdown_duration,
            y: plotData.p_human_power_grabs_reduction,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: humanPowerGrabsColor,
                width: 2
            },
            name: 'Human Power Grabs',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Slowdown duration (years)',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            automargin: true
        },
        yaxis: {
            title: 'Risk reduction',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
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
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    // Clear loading indicator before plotting
    const plotContainer = document.getElementById('agreementRiskPlot');
    if (plotContainer) {
        plotContainer.innerHTML = '';
    }

    Plotly.newPlot('agreementRiskPlot', traces, layout, {displayModeBar: false, responsive: true});
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
    module.exports = { loadAgreementSection, updateAgreementSection, plotAgreementRiskOverTime, showAgreementRiskLoadingIndicator };
}
