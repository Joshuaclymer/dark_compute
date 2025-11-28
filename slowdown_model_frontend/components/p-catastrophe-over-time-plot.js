// P(Catastrophe) Over Time Plot Component JavaScript

/**
 * Plot P(catastrophe) vs Duration of Slowdown
 *
 * This plot shows how P(catastrophe) evolves over time during the slowdown period.
 * It displays:
 * - P(AI Takeover) over time
 * - P(Human Power Grabs) over time
 * - Combined P(Catastrophe) over time
 *
 * X-axis: Duration of slowdown (years since agreement)
 * Y-axis: Probability
 */
function plotPCatastropheOverTime(data) {
    const plotData = data.p_catastrophe_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('pCatastropheOverTimePlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">No P(catastrophe) over time data available</p>';
        }
        return;
    }

    const traces = [
        // P(AI Takeover) line
        {
            x: plotData.slowdown_duration,
            y: plotData.p_ai_takeover,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#c0392b',
                width: 2
            },
            name: 'P(AI Takeover)',
            hovertemplate: 'Duration: %{x:.2f} years<br>P(AI Takeover): %{y:.3f}<extra></extra>'
        },
        // P(Human Power Grabs) line
        {
            x: plotData.slowdown_duration,
            y: plotData.p_human_power_grabs,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#8e44ad',
                width: 2
            },
            name: 'P(Human Power Grabs)',
            hovertemplate: 'Duration: %{x:.2f} years<br>P(Power Grabs): %{y:.3f}<extra></extra>'
        },
        // Combined P(Catastrophe) line (thicker, main focus)
        {
            x: plotData.slowdown_duration,
            y: plotData.p_catastrophe,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#2c3e50',
                width: 3
            },
            name: 'P(Catastrophe)',
            hovertemplate: 'Duration: %{x:.2f} years<br>P(Catastrophe): %{y:.3f}<extra></extra>'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Duration of Slowdown (years since agreement)',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            gridcolor: '#e0e0e0',
            showgrid: true,
            zeroline: true,
            zerolinecolor: '#ccc'
        },
        yaxis: {
            title: 'Probability',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            range: [0, 1],
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
        margin: { l: 70, r: 40, t: 40, b: 70 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa',
        annotations: []
    };

    // Add annotation for SC milestone if it exists
    if (plotData.sc_time && plotData.agreement_year) {
        const scDuration = plotData.sc_time - plotData.agreement_year;
        // Only add annotation if SC milestone is in the visible range
        if (scDuration > 0 && plotData.slowdown_duration[0] <= scDuration) {
            layout.shapes = [{
                type: 'line',
                x0: 0,
                x1: 0,
                y0: 0,
                y1: 1,
                xref: 'x',
                yref: 'paper',
                line: {
                    color: '#27ae60',
                    width: 2,
                    dash: 'dash'
                }
            }];
            layout.annotations.push({
                x: 0.02,
                y: 0.5,
                xref: 'paper',
                yref: 'paper',
                text: 'SC milestone<br>(handoff starts)',
                showarrow: false,
                font: {
                    size: 11,
                    color: '#27ae60'
                },
                bgcolor: 'rgba(255, 255, 255, 0.8)',
                borderpad: 4
            });
        }
    }

    Plotly.newPlot('pCatastropheOverTimePlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotPCatastropheOverTime };
}
