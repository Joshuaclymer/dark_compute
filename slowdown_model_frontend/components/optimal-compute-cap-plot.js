// Optimal Compute Cap Over Time Plot Component JavaScript

/**
 * Plot Optimal Compute Cap vs Duration of Slowdown
 *
 * This plot shows the optimal compute cap over time during the slowdown period.
 * The optimal cap is defined as the (1 - P_catastrophe) quantile of the PRC
 * covert compute distribution at each time point.
 *
 * X-axis: Duration of slowdown (years since agreement)
 * Y-axis: Compute (H100-equivalents)
 */
function plotOptimalComputeCapOverTime(data) {
    const plotData = data.optimal_compute_cap_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        const container = document.getElementById('optimalComputeCapPlot');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">No optimal compute cap data available</p>';
        }
        return;
    }

    const traces = [
        // Optimal compute cap line
        {
            x: plotData.slowdown_duration,
            y: plotData.optimal_compute_cap,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#2980b9',
                width: 3
            },
            name: 'Optimal Compute Cap',
            hovertemplate: 'Duration: %{x:.2f} years<br>Optimal Cap: %{y:,.0f} H100e<br>Quantile: %{customdata:.1%}<extra></extra>',
            customdata: plotData.optimal_quantile
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
            title: 'Compute (H100-equivalents)',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            gridcolor: '#e0e0e0',
            showgrid: true,
            tickformat: ',.0f'
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
        margin: { l: 80, r: 40, t: 40, b: 70 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa',
        annotations: []
    };

    Plotly.newPlot('optimalComputeCapPlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotOptimalComputeCapOverTime };
}
