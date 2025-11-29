// Risk Breakdown Plot Component JavaScript

/**
 * Plot risk curves for slowdown vs no slowdown comparison
 * Shows three plots: Risk without slowdown - Risk with slowdown = Risk reduction
 */

// Colors consistent with dark compute main page
const riskColors = {
    catastrophe: '#5B8DBE',     // Blue for combined P(Domestic Takeover)
    aiTakeover: '#E8A863',      // Orange for AI Takeover
    humanPowerGrabs: '#5AA89B'  // Turquoise green for Human Power Grabs
};

// Common layout settings for risk breakdown plots
function getRiskPlotLayout(showLegend = false) {
    return {
        xaxis: {
            title: 'Slowdown duration (years)',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        yaxis: {
            title: 'Probability',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            range: [0, 1],
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        showlegend: showLegend,
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#ddd',
            borderwidth: 1,
            font: { size: 9 }
        },
        hovermode: 'closest',
        margin: { l: 45, r: 10, t: 10, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa'
    };
}

/**
 * Plot risk curves WITHOUT slowdown
 */
function plotRiskNoSlowdown(data) {
    const plotData = data.risk_reduction_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('riskNoSlowdownPlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">No data available</p>';
        }
        return;
    }

    const traces = [
        {
            x: plotData.slowdown_duration,
            y: plotData.p_catastrophe_no_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.catastrophe, width: 2.5 },
            name: 'Domestic Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(Domestic Takeover): %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_ai_takeover_no_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.aiTakeover, width: 1.5 },
            name: 'AI Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(AI Takeover): %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_human_power_grabs_no_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.humanPowerGrabs, width: 1.5 },
            name: 'Human Power Grabs',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(Power Grabs): %{y:.3f}<extra></extra>'
        }
    ];

    Plotly.newPlot('riskNoSlowdownPlot', traces, getRiskPlotLayout(true), {displayModeBar: false, responsive: true});
}

/**
 * Plot risk curves WITH slowdown
 */
function plotRiskWithSlowdown(data) {
    const plotData = data.risk_reduction_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('riskWithSlowdownPlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">No data available</p>';
        }
        return;
    }

    const traces = [
        {
            x: plotData.slowdown_duration,
            y: plotData.p_catastrophe_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.catastrophe, width: 2.5 },
            name: 'Domestic Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(Domestic Takeover): %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_ai_takeover_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.aiTakeover, width: 1.5 },
            name: 'AI Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(AI Takeover): %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_human_power_grabs_slowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.humanPowerGrabs, width: 1.5 },
            name: 'Human Power Grabs',
            hovertemplate: 'Duration: %{x:.1f} years<br>P(Power Grabs): %{y:.3f}<extra></extra>'
        }
    ];

    Plotly.newPlot('riskWithSlowdownPlot', traces, getRiskPlotLayout(false), {displayModeBar: false, responsive: true});
}

/**
 * Plot risk reduction (the result of subtraction)
 */
function plotRiskReductionResult(data) {
    const plotData = data.risk_reduction_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('riskReductionResultPlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">No data available</p>';
        }
        return;
    }

    const traces = [
        {
            x: plotData.slowdown_duration,
            y: plotData.risk_reduction,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.catastrophe, width: 2.5 },
            name: 'Domestic Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_ai_takeover_reduction,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.aiTakeover, width: 1.5 },
            name: 'AI Takeover',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        },
        {
            x: plotData.slowdown_duration,
            y: plotData.p_human_power_grabs_reduction,
            type: 'scatter',
            mode: 'lines',
            line: { color: riskColors.humanPowerGrabs, width: 1.5 },
            name: 'Human Power Grabs',
            hovertemplate: 'Duration: %{x:.1f} years<br>Risk Reduction: %{y:.3f}<extra></extra>'
        }
    ];

    // Modified layout for risk reduction - y-axis can go negative
    const layout = getRiskPlotLayout(false);
    layout.yaxis.title = 'Risk reduction';
    layout.yaxis.range = undefined; // Auto-range for risk reduction
    layout.yaxis.zeroline = true;
    layout.yaxis.zerolinecolor = '#999';
    layout.yaxis.zerolinewidth = 1;

    Plotly.newPlot('riskReductionResultPlot', traces, layout, {displayModeBar: false, responsive: true});
}

/**
 * Main function to plot all three risk breakdown plots
 */
function plotRiskBreakdown(data) {
    plotRiskNoSlowdown(data);
    plotRiskWithSlowdown(data);
    plotRiskReductionResult(data);
}

// Legacy function name for compatibility
function plotPCatastropheOverTime(data) {
    plotRiskBreakdown(data);
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotPCatastropheOverTime, plotRiskBreakdown, plotRiskNoSlowdown, plotRiskWithSlowdown, plotRiskReductionResult };
}
