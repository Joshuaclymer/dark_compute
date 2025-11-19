// JavaScript for Covert Fab and Dark Compute Model sections
function plotTimeSeries(data) {
    if (!data.dark_compute_model || !data.covert_fab) return;

    const dcm = data.dark_compute_model;
    const years = dcm.years;

    // Better color scheme
    const lrColor = '#5B8DBE'; // Muted blue
    const computeColor = '#9B7BB3'; // Purple

    // Create traces for individual simulations (show first 100 for performance)
    const individualsToShow = Math.min(100, data.covert_fab.lr_combined.individual ? data.covert_fab.lr_combined.individual.length : 0);

    const individualLRs = (data.covert_fab.lr_combined.individual || []).slice(0, individualsToShow).map((lrs, idx) => ({
        x: years,
        y: lrs,
        type: 'scatter',
        mode: 'lines',
        line: { color: lrColor, width: 0.8 },
        opacity: 0.08,
        showlegend: false,
        hoverinfo: 'skip'
    }));

    const individualH100e = (dcm.covert_fab_flow.individual || []).slice(0, individualsToShow).map((h100e, idx) => ({
        x: years,
        y: h100e,
        type: 'scatter',
        mode: 'lines',
        line: { color: computeColor, width: 0.8 },
        opacity: 0.08,
        showlegend: false,
        hoverinfo: 'skip',
        yaxis: 'y2'
    }));

    // Median and percentile traces
    const traces = [
        ...individualLRs,
        {
            x: years,
            y: data.covert_fab.lr_combined.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
        },
        {
            x: years,
            y: data.covert_fab.lr_combined.p75,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: lrColor + '30',
            line: { color: 'transparent' },
            name: '25th-75th %ile (LR)',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>'
        },
        {
            x: years,
            y: data.covert_fab.lr_combined.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: lrColor, width: 3 },
            name: 'Median Evidence of Covert Fab',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>'
        },
        ...individualH100e,
        {
            x: years,
            y: dcm.covert_fab_flow.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
        },
        {
            x: years,
            y: dcm.covert_fab_flow.p75,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: computeColor + '60',
            line: { color: 'transparent' },
            name: '25th-75th %ile (H100e)',
            yaxis: 'y2',
            hovertemplate: 'H100e: %{y:,.0f}<extra></extra>'
        },
        {
            x: years,
            y: dcm.covert_fab_flow.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: computeColor, width: 3 },
            name: 'Median H100e',
            yaxis: 'y2',
            hovertemplate: 'H100e: %{y:,.0f}<extra></extra>'
        }
    ];

    // Calculate max value from 75th percentile for y-axis range
    const maxP75 = Math.max(...dcm.covert_fab_flow.p75);
    const yAxisMax = maxP75 * 1.5;

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true
        },
        yaxis: {
            title: 'Evidence of Covert Fab (Likelihood Ratio)',
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: lrColor },
            type: 'log',
            side: 'left',
            automargin: true
        },
        yaxis2: {
            title: 'H100e Produced by Fab',
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: computeColor },
            overlaying: 'y',
            side: 'right',
            automargin: true,
            tickformat: '.2s',
            range: [0, yAxisMax]
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: -0.25,
            xanchor: 'left',
            yanchor: 'top',
            orientation: 'h',
            font: { size: 10 }
        },
        hovermode: 'x unified',
        margin: { l: 50, r: 10, t: 10, b: 85, pad: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('timeSeriesPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('timeSeriesPlot'), 50);
}


function plotH100YearsTimeSeries(data) {
    if (!data.dark_compute_model) return;

    const dcm = data.dark_compute_model;
    const years = dcm.years;

    // Colors - use turquoise green for H100-years (like "other intelligence")
    const h100YearsColor = '#5AA89B'; // Turquoise green
    const lrColor = '#E57373'; // Red for likelihood ratio

    // Create traces - median lines with confidence intervals
    const traces = [
        // H100-years confidence interval
        {
            x: years.concat(years.slice().reverse()),
            y: dcm.h100_years.p75.concat(dcm.h100_years.p25.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(90, 168, 155, 0.2)',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // H100-years median
        {
            x: years,
            y: dcm.h100_years.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: h100YearsColor, width: 3 },
            name: 'H100-Years',
            yaxis: 'y',
            hovertemplate: 'H100-Years: %{y:.1f}<extra></extra>'
        },
        // Cumulative LR confidence interval
        {
            x: years.concat(years.slice().reverse()),
            y: dcm.cumulative_lr.p75.concat(dcm.cumulative_lr.p25.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(229, 115, 115, 0.2)',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
        },
        // LR median
        {
            x: years,
            y: dcm.cumulative_lr.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: lrColor, width: 3 },
            name: 'Likelihood Ratio',
            yaxis: 'y2',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true
        },
        yaxis: {
            title: 'H100-Years of Computation',
            titlefont: { size: 13, color: h100YearsColor },
            tickfont: { size: 10, color: h100YearsColor },
            side: 'left',
            automargin: true
        },
        yaxis2: {
            title: 'Likelihood Ratio',
            titlefont: { size: 13, color: lrColor },
            tickfont: { size: 10, color: lrColor },
            overlaying: 'y',
            side: 'right',
            type: 'log',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: -0.25,
            xanchor: 'left',
            yanchor: 'top',
            orientation: 'h',
            font: { size: 10 }
        },
        hovermode: 'x unified',
        margin: { l: 50, r: 50, t: 10, b: 85, pad: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('h100YearsTimeSeriesPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('h100YearsTimeSeriesPlot'), 50);
}

function plotComputeCcdf(data) {
    if (!data.covert_fab || !data.covert_fab.compute_ccdfs) {
        document.getElementById('computeCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    // Use likelihood ratios from backend
    const likelihoodRatios = data.likelihood_ratios || [1, 3, 5];
    const colors = ['#9B7BB3', '#5B8DBE', '#5AA89B'];  // Purple, Blue, Blue-green

    const traces = [];
    const thresholds = likelihoodRatios.map((lr, index) => {
        return {
            value: lr,  // Use LR directly as the key
            label: `>${lr}x update`,
            color: colors[index % colors.length]
        };
    });

    // Reverse thresholds for legend order (highest to lowest)
    const thresholdsReversed = [...thresholds].reverse();

    for (const threshold of thresholdsReversed) {
        const ccdf = data.covert_fab.compute_ccdfs[threshold.value];
        if (ccdf && ccdf.length > 0) {
            traces.push({
                x: ccdf.map(d => d.x),
                y: ccdf.map(d => d.y),
                type: 'scatter',
                mode: 'lines',
                line: { color: threshold.color, width: 2 },
                name: `"Detection" = ${threshold.label}`,
                hovertemplate: 'H100e: %{x:.0f}<br>P(≥x): %{y:.3f}<extra></extra>'
            });
        }
    }

    const layout = {
        xaxis: {
            title: "H100e Produced by Covert Fab Before 'Detection'",
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            type: 'log',
            automargin: true
        },
        yaxis: {
            title: 'P(covert compute > x)',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            range: [0, 1],
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.98,
            y: 0.98,
            xanchor: 'right',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 50, r: 10, t: 10, b: 65, pad: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('computeCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('computeCcdfPlot'), 50);
}

function plotProjectH100YearsCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.h100_years_ccdf) {
        document.getElementById('projectH100YearsCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    // Get prior probability from the data
    const priorProb = data.p_project_exists || 0.1;
    const priorOdds = priorProb / (1 - priorProb);

    // Use likelihood ratios from backend
    const likelihoodRatios = data.likelihood_ratios || [1, 2, 5];
    const colors = ['#9B7BB3', '#5B8DBE', '#5AA89B'];  // Purple, Blue, Blue-green

    const traces = [];
    const thresholds = likelihoodRatios.map((lr, index) => {
        const posteriorOdds = priorOdds * lr;
        const posteriorProb = posteriorOdds / (1 + posteriorOdds);
        return {
            value: posteriorProb,
            label: `>${lr}x update`,
            color: colors[index % colors.length]
        };
    });


    // Reverse thresholds for legend order (highest to lowest)
    const thresholdsReversed = [...thresholds].reverse();

    for (const threshold of thresholdsReversed) {
        const ccdf = data.dark_compute_model.h100_years_ccdf[threshold.value];
        if (ccdf && ccdf.length > 0) {
            traces.push({
                x: ccdf.map(d => d.x),
                y: ccdf.map(d => d.y),
                type: 'scatter',
                mode: 'lines',
                line: { color: threshold.color, width: 2 },
                name: `"Detection" = ${threshold.label}`,
                hovertemplate: 'H100-years: %{x:.0f}<br>P(≥x): %{y:.3f}<extra></extra>'
            });
        }
    }

    const layout = {
        xaxis: {
            title: "H100-Years Before 'Detection'",
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            type: 'log',
            automargin: true,
            showline: false,
            mirror: false
        },
        yaxis: {
            title: 'P(H100-years > x)',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            range: [0, 1],
            automargin: true,
            showline: false,
            mirror: false
        },
        showlegend: true,
        legend: {
            x: 0.98,
            y: 0.98,
            xanchor: 'right',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 50, r: 50, t: 10, b: 65, pad: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('projectH100YearsCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('projectH100YearsCcdfPlot'), 50);
}

function updateDashboard(data) {
    // Find the median simulation by H100e production
    const h100eValues = data.covert_fab?.individual_h100e_before_detection || [];
    const timeValues = data.covert_fab?.individual_time_before_detection || [];
    const nodeValues = data.covert_fab?.individual_process_node || [];
    const energyValues = data.covert_fab?.individual_energy_before_detection || [];

    if (h100eValues.length > 0) {
        // Sort by H100e and find median simulation
        const indexed = h100eValues.map((h100e, idx) => ({ h100e, idx }));
        indexed.sort((a, b) => a.h100e - b.h100e);
        const idx50 = Math.floor(indexed.length * 0.5);
        const sim80th = indexed[idx50];

        // Get all values from this same simulation
        const h100e80th = sim80th.h100e;
        const time80th = timeValues[sim80th.idx];
        const node80th = nodeValues[sim80th.idx];
        const energy80th = energyValues[sim80th.idx];

        // Display H100e and energy combined
        const rounded = Math.round(h100e80th / 100000) * 100000;
        let h100eText;
        if (rounded >= 1000000) {
            h100eText = `${(rounded / 1000000).toFixed(1)}M H100e`;
        } else if (rounded >= 1000) {
            h100eText = `${(rounded / 1000).toFixed(0)}K H100e`;
        } else {
            h100eText = `${rounded.toFixed(0)} H100e`;
        }

        let energyText;
        if (energy80th >= 1) {
            energyText = `${energy80th.toFixed(1)} GW`;
        } else if (energy80th >= 0.001) {
            energyText = `${(energy80th * 1000).toFixed(0)} MW`;
        } else {
            energyText = `${(energy80th * 1000).toFixed(1)} MW`;
        }

        document.getElementById('dashboard80thCombined').innerHTML =
            `${h100eText}<br><span style="font-size: 24px; color: #666;">${energyText}</span>`;

        // Display time
        document.getElementById('dashboard80thTime').textContent = time80th.toFixed(1);

        // Display process node
        document.getElementById('dashboard80thNode').textContent = node80th;

        // Display probability of fab being built
        const probFabBuilt = data.prob_fab_built !== undefined ? (data.prob_fab_built * 100).toFixed(1) + '%' : '--';
        document.getElementById('dashboardProbFabBuilt').textContent = probFabBuilt;

        // Update detection label with highest LR value
        const likelihoodRatios = data.likelihood_ratios || [1, 3, 5];
        const highestLR = likelihoodRatios[likelihoodRatios.length - 1];
        document.getElementById('dashboardDetectionLabel').textContent =
            `Detection means ≥${highestLR}x update`;
    }

    // Populate dark compute project dashboard
    const projectH100eValues = data.dark_compute_model?.individual_project_h100e_before_detection || [];
    const projectEnergyValues = data.dark_compute_model?.individual_project_energy_before_detection || [];
    const projectTimeValues = data.dark_compute_model?.individual_project_time_before_detection || [];
    const projectH100YearsValues = data.dark_compute_model?.individual_project_h100_years_before_detection || [];

    if (projectH100eValues.length > 0) {
        // Sort by H100-years and find 80th percentile simulation
        const projectIndexed = projectH100YearsValues.map((h100years, idx) => ({ h100years, idx }));
        projectIndexed.sort((a, b) => a.h100years - b.h100years);
        const projectIdx80 = Math.floor(projectIndexed.length * 0.8);
        const projectSim80th = projectIndexed[projectIdx80];

        // Get all values from this same simulation
        const projectH100Years80th = projectSim80th.h100years;
        const projectH100e80th = projectH100eValues[projectSim80th.idx];
        const projectEnergy80th = projectEnergyValues[projectSim80th.idx];
        const projectTime80th = projectTimeValues[projectSim80th.idx];

        // Display H100-years
        const h100YearsRounded = Math.round(projectH100Years80th / 100000) * 100000;
        if (h100YearsRounded >= 1000000) {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `~${(h100YearsRounded / 1000000).toFixed(1)}M H100-years`;
        } else if (h100YearsRounded >= 1000) {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `~${(h100YearsRounded / 1000).toFixed(0)}K H100-years`;
        } else {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `~${h100YearsRounded.toFixed(0)} H100-years`;
        }

        // Display H100e and energy combined
        const projectRounded = Math.round(projectH100e80th / 100000) * 100000;
        let projectH100eText;
        if (projectRounded >= 1000000) {
            projectH100eText = `${(projectRounded / 1000000).toFixed(1)}M H100e`;
        } else if (projectRounded >= 1000) {
            projectH100eText = `${(projectRounded / 1000).toFixed(0)}K H100e`;
        } else {
            projectH100eText = `${projectRounded.toFixed(0)} H100e`;
        }

        let projectEnergyText;
        if (projectEnergy80th >= 1) {
            projectEnergyText = `${projectEnergy80th.toFixed(1)} GW`;
        } else if (projectEnergy80th >= 0.001) {
            projectEnergyText = `${(projectEnergy80th * 1000).toFixed(0)} MW`;
        } else {
            projectEnergyText = `${(projectEnergy80th * 1000).toFixed(1)} MW`;
        }

        document.getElementById('dashboardProject80thH100eCombined').innerHTML =
            `${projectH100eText}<br><span style="font-size: 24px; color: #666;">${projectEnergyText}</span>`;

        // Display time
        document.getElementById('dashboardProject80thTime').textContent = projectTime80th.toFixed(1);
    }
}