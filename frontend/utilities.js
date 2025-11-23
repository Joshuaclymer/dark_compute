// Utility plotting functions shared across sections

// ============================================================================
// CENTRALIZED HOVER CONFIGURATION
// ============================================================================
// All hover effects should use these standardized configurations to ensure
// consistency across the entire application. To modify hover behavior globally,
// update these configuration objects.
// ============================================================================

const HoverConfig = {
    // For time series plots with x-axis (e.g., line charts over time)
    // This creates the unified vertical hover line effect
    timeSeries: {
        hovermode: 'x unified'
    },

    // For scatter plots, CCDF plots, and other non-time-series visualizations
    scatter: {
        hovermode: 'closest'
    },

    // For bar charts and categorical data
    bar: {
        hovermode: 'closest'
    },

    // Helper function to get hover template that matches the unified style
    // Use this for traces that need custom hover templates
    getTemplate: (label, valueFormat = '.2f') => {
        return `${label}: %{y:${valueFormat}}<extra></extra>`;
    },

    // Helper for XY hover templates (e.g., CCDF plots)
    getXYTemplate: (xLabel, yLabel, xFormat = '.2f', yFormat = '.3f') => {
        return `${xLabel}: %{x:${xFormat}}<br>${yLabel}: %{y:${yFormat}}<extra></extra>`;
    },

    // Mark traces that should not show hover info (e.g., shaded regions, percentile bands)
    skipHover: {
        hoverinfo: 'skip'
    }
};

function plotPDF(divId, values, color, xAxisLabel, nbins = 30, logScale = false, logMin = null, logMax = null, title = null, pattern = null) {
    // Create histogram/PDF from values

    const marker = { color: color, line: { width: 0.5, color: 'white' } };

    // Add pattern if specified
    if (pattern) {
        marker.pattern = pattern;
    }

    const trace = {
        x: values,
        type: 'histogram',
        histnorm: 'probability density',
        marker: marker,
        hovertemplate: '%{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
    };

    // For log scale, manually specify bin edges
    if (logScale) {
        // Use custom range if provided, otherwise default to 0.1 to 10
        const minVal = logMin !== null ? logMin : 0.1;
        const maxVal = logMax !== null ? logMax : 10;
        const numBins = 12;
        const logMinVal = Math.log10(minVal);
        const logMaxVal = Math.log10(maxVal);
        const binEdges = [];
        for (let i = 0; i <= numBins; i++) {
            binEdges.push(Math.pow(10, logMinVal + i * (logMaxVal - logMinVal) / numBins));
        }

        // Manually bin the data
        const binCounts = new Array(binEdges.length - 1).fill(0);
        values.forEach(v => {
            // Find which bin this value belongs to
            let binned = false;
            for (let i = 0; i < binEdges.length - 2; i++) {
                if (v >= binEdges[i] && v < binEdges[i + 1]) {
                    binCounts[i]++;
                    binned = true;
                    break;
                }
            }
            // Last bin captures all values >= second-to-last edge (making it a "+" bin)
            if (!binned) {
                binCounts[binCounts.length - 1]++;
            }
        });

        // Calculate bin centers for plotting
        const binCenters = [];
        for (let i = 0; i < binEdges.length - 1; i++) {
            binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
        }

        // Calculate probability mass (count / total) for better visualization with open-ended bins
        // Using probability instead of density makes the bar height proportional to the actual probability mass
        const probabilities = binCounts.map(count => count / values.length);


        // Create bar trace instead of histogram
        trace.type = 'bar';
        trace.x = binCenters;
        trace.y = probabilities;
        delete trace.histnorm;
        trace.width = binEdges.map((edge, i) => i < binEdges.length - 1 ? binEdges[i + 1] - binEdges[i] : 0).slice(0, -1);
        trace.hovertemplate = 'LR: %{x:.2f}<br>Probability: %{y:.3f}<extra></extra>';
    } else {
        trace.nbinsx = nbins;
    }

    const xaxisConfig = {
        title: xAxisLabel,
        titlefont: { size: 10 },
        automargin: true,
        tickfont: { size: 9 }
    };

    const yaxisConfig = {
        title: {
            text: logScale ? 'Probability' : 'Density',
            standoff: 15
        },
        titlefont: { size: 10 },
        tickfont: { size: 9 }
    };

    if (logScale) {
        xaxisConfig.type = 'log';
        // Manually specify tick values for better visibility
        // Use the actual max value from logMax parameter
        const maxTickVal = logMax !== null ? logMax : 10;
        const maxTickText = maxTickVal.toString() + '+';
        xaxisConfig.tickvals = [0.1, 0.2, 0.5, 1, 2, maxTickVal];
        xaxisConfig.ticktext = ['0.1', '0.2', '0.5', '1', '2', maxTickText];
        // Don't force range - let it adjust based on data
        xaxisConfig.autorange = true;
    } else {
        // Use k/M formatting for non-log scale plots
        xaxisConfig.tickformat = '.2s';
    }

    const layout = {
        xaxis: xaxisConfig,
        yaxis: {
            ...yaxisConfig,
            automargin: true
        },
        showlegend: false,
        margin: { l: 55, r: 0, t: title ? 30 : 0, b: 60 },
        hovermode: 'closest',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true
    };

    if (title) {
        layout.title = {
            text: `<b>${title}</b>`,
            font: { size: 12 },
            x: 0.5,
            xanchor: 'center'
        };
    }

    Plotly.newPlot(divId, [trace], layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize(divId), 50);
}

function plotMedianWithPercentiles(divId, data, years, color, yAxisLabel = '', yAxisRange = null, logScale = false, showLegend = false) {
    // Plot median with 25th-75th percentile bands
    const traces = [];

    // Add individual simulation lines if provided
    if (data.individual) {
        const individualsToPlot = data.individual.slice(0, 100);
        for (let i = 0; i < individualsToPlot.length; i++) {
            traces.push({
                x: years,
                y: individualsToPlot[i],
                type: 'scatter',
                mode: 'lines',
                line: { color: color, width: 0.5 },
                opacity: 0.15,
                showlegend: false,
                hoverinfo: 'skip'
            });
        }
    }

    // Add percentile bands
    traces.push(
        {
            x: years,
            y: data.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: color, width: 1.5, dash: 'dot' },
            opacity: 0.6,
            showlegend: false,
            hoverinfo: 'skip',
            name: '25th percentile'
        },
        {
            x: years,
            y: data.p75,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: color + '60',
            line: { color: color, width: 1.5, dash: 'dot' },
            opacity: 0.7,
            showlegend: showLegend,
            hoverinfo: 'skip',
            name: '25th-75th %ile'
        },
        {
            x: years,
            y: data.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: color, width: 3 },
            showlegend: showLegend,
            hovertemplate: 'Year: %{x:.1f}<br>Value: %{y:.2f}<extra></extra>',
            name: 'Median'
        }
    );

    const yaxisConfig = {
        title: yAxisLabel,
        titlefont: { size: 10 },
        tickfont: { size: 9 },
        automargin: true
    };

    // Add log scale if specified
    if (logScale) {
        yaxisConfig.type = 'log';
    }

    // Add range if specified
    if (yAxisRange !== null) {
        yaxisConfig.range = yAxisRange;
    }

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true,
            range: [years[0] - 0.1, years[years.length - 1] + 0.1],
            autorange: false,
            fixedrange: true
        },
        yaxis: yaxisConfig,
        showlegend: showLegend,
        legend: {
            x: 0.98,
            y: 0.98,
            xanchor: 'right',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1,
            font: { size: 10 }
        },
        margin: { l: 50, r: 10, t: 10, b: 55, pad: 10 },
        hovermode: 'x'
    };

    Plotly.newPlot(divId, traces, layout, {displayModeBar: false, responsive: true});
    // Force resize after a small delay
    setTimeout(() => Plotly.Plots.resize(divId), 50);
}

function plotBarByProcessNode(divId, values, processNodes, color) {
    // Group by process node and compute per wafer value
    const nodeValueCounts = {};
    values.forEach((v, idx) => {
        const rounded = Math.round(v * 1000) / 1000; // Round to 3 decimal places
        const node = processNodes[idx];
        const key = `${rounded}|${node}`;
        nodeValueCounts[key] = (nodeValueCounts[key] || 0) + 1;
    });

    // Parse and sort by value
    const entries = Object.entries(nodeValueCounts).map(([key, count]) => {
        const [value, node] = key.split('|');
        return { value: parseFloat(value), node, count };
    });
    entries.sort((a, b) => a.value - b.value);

    // Convert counts to proportions
    const totalCount = values.length;
    const labels = entries.map(e => `${e.value.toFixed(2)}x (${e.node})`);
    const proportions = entries.map(e => e.count / totalCount);

    const trace = {
        x: labels,
        y: proportions,
        type: 'bar',
        marker: {
            color: color,
            line: { width: 1, color: 'white' }
        },
        textposition: 'auto',
        hovertemplate: '%{x}<br>Probability: %{y:.3f}<extra></extra>'
    };

    const layout = {
        xaxis: {
            title: '',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true,
            type: 'category'
        },
        yaxis: {
            title: 'Probability',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true
        },
        showlegend: false,
        margin: { l: 50, r: 10, t: 10, b: 65, pad: 10 },
        hovermode: 'closest',
        bargap: 0.2
    };

    Plotly.newPlot(divId, [trace], layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize(divId), 50);
}

function plotProportionOperational(divId, data, years, color, yAxisLabel = '') {
    // Use the proportion directly (proportion of simulations where fab is operational)
    const proportions = data.proportion;

    const trace = {
        x: years,
        y: proportions,
        type: 'scatter',
        mode: 'lines',
        line: { color: color, width: 2 },
        fill: 'tozeroy',
        fillcolor: color + '20',
        hovertemplate: 'Year: %{x:.1f}<br>Proportion: %{y:.2f}<extra></extra>'
    };

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true,
            range: [years[0] - 0.1, years[years.length - 1] + 0.1],
            autorange: false,
            fixedrange: true
        },
        yaxis: {
            title: yAxisLabel,
            range: [0, 1],
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true
        },
        showlegend: false,
        margin: { l: 50, r: 10, t: 10, b: 55, pad: 10 },
        hovermode: 'x'
    };

    Plotly.newPlot(divId, [trace], layout, {displayModeBar: false, responsive: true});
    // Force resize after a small delay
    setTimeout(() => Plotly.Plots.resize(divId), 50);
}

function plotCategoricalFrequency(divId, values, categories, color) {
    // Count frequencies for each category
    const frequencies = categories.map(cat =>
        values.filter(v => v === cat.value).length
    );

    // Convert to proportions
    const total = values.length;
    const proportions = frequencies.map(f => f / total);

    const trace = {
        x: categories.map(cat => cat.label),
        y: proportions,
        type: 'bar',
        marker: {
            color: color,
            line: { width: 1, color: 'white' }
        },
        textposition: 'auto',
        hovertemplate: '%{x}<br>Probability: %{y:.3f}<extra></extra>'
    };

    const layout = {
        xaxis: {
            title: '',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true,
            type: 'category'
        },
        yaxis: {
            title: 'Probability',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true
        },
        showlegend: false,
        margin: { l: 50, r: 10, t: 10, b: 65, pad: 10 },
        hovermode: 'closest',
        bargap: 0.2
    };

    Plotly.newPlot(divId, [trace], layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize(divId), 50);
}
