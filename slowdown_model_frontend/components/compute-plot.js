// Compute Over Time Plot Component JavaScript

function plotCovertCompute(data) {
    // Use combined covert compute data (pre-agreement PRC + post-agreement covert)
    const combinedData = data.combined_covert_compute;
    const largestCompanyData = data.largest_company_compute;
    const prcNoSlowdownData = data.prc_no_slowdown_compute;
    const proxyProjectData = data.proxy_project_compute;

    if (!combinedData || !combinedData.years || !combinedData.median) {
        document.getElementById('covertComputePlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">No covert compute data available. Please run a simulation first.</p>';
        return;
    }

    const years = combinedData.years;
    const median = combinedData.median;
    const agreement_year = data.agreement_year;

    // Create traces for the plot
    const traces = [
        // PRC Covert Compute line (with slowdown)
        {
            x: years,
            y: median,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#8B4513',
                width: 3
            },
            name: 'PRC Covert Compute',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        }
    ];

    // Add PRC no-slowdown compute if available
    if (prcNoSlowdownData && prcNoSlowdownData.years && prcNoSlowdownData.median) {
        traces.push({
            x: prcNoSlowdownData.years,
            y: prcNoSlowdownData.median,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#D2691E',
                width: 3,
                dash: 'dash'
            },
            name: 'PRC Compute (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // Add Proxy Project compute: only from agreement year onwards
    // This represents the compute cap based on PRC covert compute estimates
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

    // Add largest company compute (no slowdown baseline) - full trajectory
    if (largestCompanyData && largestCompanyData.years && largestCompanyData.compute) {
        traces.push({
            x: largestCompanyData.years,
            y: largestCompanyData.compute,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#2A623D',
                width: 3,
                dash: 'dash'
            },
            name: 'Largest U.S. Company (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // Add a vertical line at agreement year
    const shapes = [{
        type: 'line',
        x0: agreement_year,
        x1: agreement_year,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: {
            color: 'rgba(100, 100, 100, 0.5)',
            width: 2,
            dash: 'dash'
        }
    }];

    // Add annotation for the agreement year
    const annotations = [{
        x: agreement_year,
        y: 1.02,
        yref: 'paper',
        text: 'Agreement Start',
        showarrow: false,
        font: {
            size: 11,
            color: '#666'
        }
    }];

    // Determine end year from the data (last year in combined data)
    const endYear = years[years.length - 1];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            automargin: true,
            gridcolor: '#e0e0e0',
            showgrid: true,
            range: [2026, endYear]
        },
        yaxis: {
            title: 'H100-equivalents',
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
        margin: { l: 100, r: 40, t: 60, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa',
        shapes: shapes,
        annotations: annotations
    };

    Plotly.newPlot('covertComputePlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Show error message for compute plot
function showComputePlotError(message) {
    const plotContainer = document.getElementById('covertComputePlot');
    if (plotContainer) {
        plotContainer.innerHTML = '<p style="text-align: center; color: #e74c3c;">' + message + '</p>';
    }
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotCovertCompute, showComputePlotError };
}
