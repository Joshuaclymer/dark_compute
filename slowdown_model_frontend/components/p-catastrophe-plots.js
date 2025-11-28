// P(Catastrophe) Plots Component JavaScript

// Plot P(AI Takeover) curve
function plotPAiTakeover(data) {
    const durations = data.durations;
    const pTakeover = data.p_ai_takeover;
    const anchors = data.anchor_points.ai_takeover;
    const anchorTimes = data.anchor_times;

    // Format anchor time labels
    const formatAnchorLabel = (timeYears) => {
        if (timeYears < 1) {
            const months = Math.round(timeYears * 12);
            return months === 1 ? '1 month' : `${months} months`;
        }
        return timeYears === 1 ? '1 year' : `${timeYears} years`;
    };

    const traces = [
        // Main curve
        {
            x: durations,
            y: pTakeover,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#c0392b',
                width: 3
            },
            name: 'P(AI Takeover)',
            hovertemplate: 'Duration: %{x:.2f} years<br>P(Takeover): %{y:.3f}<extra></extra>'
        },
        // Anchor points
        {
            x: [anchorTimes.t1, anchorTimes.t2, anchorTimes.t3],
            y: [anchors['t1'], anchors['t2'], anchors['t3']],
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: '#c0392b',
                size: 10,
                line: {
                    color: 'white',
                    width: 2
                }
            },
            name: 'Parameter anchors',
            hovertemplate: '%{text}<br>P(Takeover): %{y:.2f}<extra></extra>',
            text: [formatAnchorLabel(anchorTimes.t1), formatAnchorLabel(anchorTimes.t2), formatAnchorLabel(anchorTimes.t3)]
        }
    ];

    const layout = {
        xaxis: {
            title: 'Research-Speed-Adjusted Handoff Duration (years)',
            titlefont: { size: 13 },
            tickfont: { size: 11 },
            gridcolor: '#e0e0e0',
            showgrid: true,
            type: 'log',
            range: [Math.log10(anchorTimes.t1), Math.log10(20)]
        },
        yaxis: {
            title: 'P(AI Takeover)',
            titlefont: { size: 13 },
            tickfont: { size: 11 },
            range: [0, 1],
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0.98,
            y: 0.98,
            xanchor: 'right',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#ddd',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 60, r: 20, t: 20, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa'
    };

    Plotly.newPlot('pAiTakeoverPlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Plot P(Human Power Grabs) curve
function plotPHumanPowerGrabs(data) {
    const durations = data.durations;
    const pPowerGrabs = data.p_human_power_grabs;
    const anchors = data.anchor_points.human_power_grabs;
    const anchorTimes = data.anchor_times;

    // Format anchor time labels
    const formatAnchorLabel = (timeYears) => {
        if (timeYears < 1) {
            const months = Math.round(timeYears * 12);
            return months === 1 ? '1 month' : `${months} months`;
        }
        return timeYears === 1 ? '1 year' : `${timeYears} years`;
    };

    const traces = [
        // Main curve
        {
            x: durations,
            y: pPowerGrabs,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#8e44ad',
                width: 3
            },
            name: 'P(Human Power Grabs)',
            hovertemplate: 'Duration: %{x:.2f} years<br>P(Power Grabs): %{y:.3f}<extra></extra>'
        },
        // Anchor points
        {
            x: [anchorTimes.t1, anchorTimes.t2, anchorTimes.t3],
            y: [anchors['t1'], anchors['t2'], anchors['t3']],
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: '#8e44ad',
                size: 10,
                line: {
                    color: 'white',
                    width: 2
                }
            },
            name: 'Parameter anchors',
            hovertemplate: '%{text}<br>P(Power Grabs): %{y:.2f}<extra></extra>',
            text: [formatAnchorLabel(anchorTimes.t1), formatAnchorLabel(anchorTimes.t2), formatAnchorLabel(anchorTimes.t3)]
        }
    ];

    const layout = {
        xaxis: {
            title: 'Handoff Duration (years)',
            titlefont: { size: 13 },
            tickfont: { size: 11 },
            gridcolor: '#e0e0e0',
            showgrid: true,
            type: 'log',
            range: [Math.log10(anchorTimes.t1), Math.log10(20)]
        },
        yaxis: {
            title: 'P(Human Power Grabs)',
            titlefont: { size: 13 },
            tickfont: { size: 11 },
            range: [0, 1],
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0.98,
            y: 0.98,
            xanchor: 'right',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#ddd',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 60, r: 20, t: 20, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa'
    };

    Plotly.newPlot('pHumanPowerGrabsPlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Plot P(catastrophe) curves from the combined data response
function plotPCatastropheFromData(data) {
    if (!data || !data.p_catastrophe_curves) {
        console.warn('No P(catastrophe) curve data available');
        return;
    }
    plotPAiTakeover(data.p_catastrophe_curves);
    plotPHumanPowerGrabs(data.p_catastrophe_curves);
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotPAiTakeover, plotPHumanPowerGrabs, plotPCatastropheFromData };
}
