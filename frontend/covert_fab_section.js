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
            name: 'LR: 25th-75th %ile',
            legendgroup: 'lr',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>'
        },
        {
            x: years,
            y: data.covert_fab.lr_combined.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: lrColor, width: 3 },
            name: 'LR: Median',
            legendgroup: 'lr',
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
            name: 'H100e: 25th-75th %ile',
            legendgroup: 'h100e',
            yaxis: 'y2',
            hovertemplate: 'H100e: %{y:,.0f}<extra></extra>'
        },
        {
            x: years,
            y: dcm.covert_fab_flow.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: computeColor, width: 3 },
            name: 'H100e: Median',
            legendgroup: 'h100e',
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
            title: {
                text: 'Evidence of Covert Fab (LR)',
                standoff: 15
            },
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: lrColor },
            type: 'log',
            side: 'left',
            automargin: true
        },
        yaxis2: {
            title: {
                text: 'H100 equivalents (FLOPS) Produced by Fab',
                standoff: 15
            },
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
            x: 0.5,
            y: -0.25,
            xanchor: 'center',
            yanchor: 'top',
            orientation: 'h',
            font: { size: 10 },
            bgcolor: 'rgba(255,255,255,0)',
            borderwidth: 0,
            tracegroupgap: 20
        },
        hovermode: 'x unified',
        margin: { l: 55, r: 55, t: 0, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('timeSeriesPlot', traces, layout, {displayModeBar: false, responsive: true});

    // Match plot heights to dashboard height after both plots are created
    setTimeout(() => {
        const dashboard = document.querySelector('#covertFabTopSection .dashboard');
        const plotContainers = document.querySelectorAll('#covertFabTopSection .plot-container');
        if (dashboard && plotContainers.length > 0) {
            const dashboardHeight = dashboard.offsetHeight;
            plotContainers.forEach(container => {
                container.style.height = dashboardHeight + 'px';
            });
            // Force resize after setting height
            setTimeout(() => {
                Plotly.Plots.resize('computeCcdfPlot');
                Plotly.Plots.resize('timeSeriesPlot');
            }, 50);
        }
    }, 150);
}


function plotH100YearsTimeSeries(data) {
    if (!data.dark_compute_model) return;

    const dcm = data.dark_compute_model;
    const years = dcm.years;

    // Colors - use blue for probability (same as "Posterior Probability of Covert Project")
    const probColor = '#5B8DBE'; // Blue for probability
    const h100YearsColor = '#5AA89B'; // Turquoise green for H100-years

    // Create traces - median lines with confidence intervals
    const traces = [
        // Posterior probability confidence interval
        {
            x: years.concat(years.slice().reverse()),
            y: dcm.posterior_prob_project.p75.concat(dcm.posterior_prob_project.p25.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(91, 141, 190, 0.2)',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // Probability median
        {
            x: years,
            y: dcm.posterior_prob_project.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: probColor, width: 3 },
            name: 'US Estimated Likelihood of PRC Covert Project',
            yaxis: 'y',
            hovertemplate: 'Probability: %{y:.3f}<extra></extra>'
        },
        // H100-years confidence interval
        {
            x: years.concat(years.slice().reverse()),
            y: dcm.h100_years.p75.concat(dcm.h100_years.p25.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(90, 168, 155, 0.2)',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
        },
        // H100-years median
        {
            x: years,
            y: dcm.h100_years.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: h100YearsColor, width: 3 },
            name: 'H100-Years',
            yaxis: 'y2',
            hovertemplate: 'H100-Years: %{y:.1f}<extra></extra>'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true
        },
        yaxis: {
            title: {
                text: 'US Estimated Likelihood<br>of a PRC Covert Project',
                standoff: 15
            },
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: probColor },
            side: 'left',
            range: [0, 1],
            automargin: true
        },
        yaxis2: {
            title: {
                text: 'H100 years of computation',
                standoff: 15
            },
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: h100YearsColor },
            overlaying: 'y',
            side: 'right',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.5,
            y: -0.25,
            xanchor: 'center',
            yanchor: 'top',
            orientation: 'h',
            font: { size: 10 },
            bgcolor: 'rgba(255,255,255,0)',
            borderwidth: 0,
            tracegroupgap: 20
        },
        hovermode: 'x unified',
        margin: { l: 55, r: 55, t: 0, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('h100YearsTimeSeriesPlot', traces, layout, {displayModeBar: false, responsive: true});

    // Match plot heights to dashboard height after both plots are created
    setTimeout(() => {
        const dashboard = document.querySelector('#darkComputeTopSection .dashboard');
        const plotContainers = document.querySelectorAll('#darkComputeTopSection .plot-container');
        if (dashboard && plotContainers.length > 0) {
            const dashboardHeight = dashboard.offsetHeight;
            plotContainers.forEach(container => {
                container.style.height = dashboardHeight + 'px';
            });
            // Force resize after setting height
            setTimeout(() => {
                Plotly.Plots.resize('projectH100YearsCcdfPlot');
                Plotly.Plots.resize('h100YearsTimeSeriesPlot');
            }, 50);
        }
    }, 150);
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
            title: "H100 equivalents (FLOPS) Produced by Covert Fab Before 'Detection'",
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
        margin: { l: 55, r: 0, t: 0, b: 60 },
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

    // Use likelihood ratios from backend
    const likelihoodRatios = data.dark_compute_model.likelihood_ratios || [1, 3, 5];
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
            automargin: true
        },
        yaxis: {
            title: 'P(H100-years > x)',
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
        margin: { l: 55, r: 0, t: 0, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('projectH100YearsCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('projectH100YearsCcdfPlot'), 50);
}

function updateDashboard(data) {
    // Update parameter display values in the text
    updateParameterDisplays();

    // Find the 80th percentile simulation by H100e production
    const h100eValues = data.covert_fab?.individual_h100e_before_detection || [];
    const timeValues = data.covert_fab?.individual_time_before_detection || [];
    const nodeValues = data.covert_fab?.individual_process_node || [];
    const energyValues = data.covert_fab?.individual_energy_before_detection || [];

    if (h100eValues.length > 0) {
        // Sort by H100e and find 80th percentile simulation
        const indexed = h100eValues.map((h100e, idx) => ({ h100e, idx }));
        indexed.sort((a, b) => a.h100e - b.h100e);
        const idx50 = Math.floor(indexed.length * 0.8);
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

        // Update the section description with the percentage
        const fabPercentageSpan = document.getElementById('fabBuiltPercentage');
        if (fabPercentageSpan) {
            fabPercentageSpan.textContent = data.prob_fab_built !== undefined ? ` (${(data.prob_fab_built * 100).toFixed(1)}% of simulations)` : '';
        }

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
        const projectIdxP80 = Math.floor(projectIndexed.length * 0.8);
        const projectSimP80 = projectIndexed[projectIdxP80];

        // Get all values from this same simulation
        const projectH100YearsMedian = projectSimP80.h100years;
        const projectH100eMedian = projectH100eValues[projectSimP80.idx];
        const projectEnergyMedian = projectEnergyValues[projectSimP80.idx];
        const projectTimeMedian = projectTimeValues[projectSimP80.idx];

        // Display H100-years
        const h100YearsRounded = Math.round(projectH100YearsMedian / 100000) * 100000;
        if (h100YearsRounded >= 1000000) {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `${(h100YearsRounded / 1000000).toFixed(1)}M H100-years`;
        } else if (h100YearsRounded >= 1000) {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `${(h100YearsRounded / 1000).toFixed(0)}K H100-years`;
        } else {
            document.getElementById('dashboardProject80thH100Years').textContent =
                `${h100YearsRounded.toFixed(0)} H100-years`;
        }

        // Display H100e and energy combined
        const projectRounded = Math.round(projectH100eMedian / 100000) * 100000;
        let projectH100eText;
        if (projectRounded >= 1000000) {
            projectH100eText = `${(projectRounded / 1000000).toFixed(1)}M H100e`;
        } else if (projectRounded >= 1000) {
            projectH100eText = `${(projectRounded / 1000).toFixed(0)}K H100e`;
        } else {
            projectH100eText = `${projectRounded.toFixed(0)} H100e`;
        }

        let projectEnergyText;
        if (projectEnergyMedian >= 1) {
            projectEnergyText = `${projectEnergyMedian.toFixed(1)} GW`;
        } else if (projectEnergyMedian >= 0.001) {
            projectEnergyText = `${(projectEnergyMedian * 1000).toFixed(0)} MW`;
        } else {
            projectEnergyText = `${(projectEnergyMedian * 1000).toFixed(1)} MW`;
        }

        document.getElementById('dashboardProject80thH100eCombined').innerHTML =
            `${projectH100eText}<br><span style="font-size: 24px; color: #666;">${projectEnergyText}</span>`;

        // Display time
        document.getElementById('dashboardProject80thTime').textContent = projectTimeMedian.toFixed(1) + ' years';

        // Update detection label with highest LR value for dark compute model
        const likelihoodRatios = data.dark_compute_model?.likelihood_ratios || data.likelihood_ratios || [1, 3, 5];
        const highestLR = likelihoodRatios[likelihoodRatios.length - 1];
        const projectDetectionLabel = document.getElementById('dashboardProjectDetectionLabel');
        if (projectDetectionLabel) {
            projectDetectionLabel.textContent = `Detection means ≥${highestLR}x update`;
        }
    }
}

function updateParameterDisplays() {
    // Helper function to format large numbers
    function formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(0) + 'K';
        }
        return num.toFixed(0);
    }

    // Update initial PRC compute stock
    const prcComputeInput = document.getElementById('total_prc_compute_stock_in_2025');
    const initialDiversionInput = document.getElementById('proportion_of_initial_chip_stock_to_divert');
    const agreementYearInput = document.getElementById('agreement_year');
    const growthRateInput = document.getElementById('annual_growth_rate_of_prc_compute_stock');

    if (prcComputeInput && initialDiversionInput && agreementYearInput && growthRateInput) {
        // Calculate PRC compute at agreement year
        const prcCompute2025 = parseFloat(prcComputeInput.value) * 1e6; // Convert from millions
        const agreementYear = parseFloat(agreementYearInput.value);
        const growthRate = parseFloat(growthRateInput.value);
        const yearsFromBase = agreementYear - 2025;

        // Update agreement year display
        const agreementYearSpan = document.getElementById('param-agreement-year');
        if (agreementYearSpan) {
            agreementYearSpan.textContent = agreementYear.toFixed(0);
            agreementYearSpan.onclick = () => {
                if (agreementYearInput) {
                    agreementYearInput.focus();
                    agreementYearInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            };
        }

        // Grow the stock to the agreement year
        const prcComputeAtAgreement = prcCompute2025 * Math.pow(growthRate, yearsFromBase);

        const diversionProportion = parseFloat(initialDiversionInput.value);
        const divertedCompute = prcComputeAtAgreement * diversionProportion;

        // Update initial PRC compute display
        const prcComputeSpan = document.getElementById('param-initial-prc-compute');
        if (prcComputeSpan) {
            prcComputeSpan.textContent = formatNumber(prcComputeAtAgreement);
            // Add click handler
            prcComputeSpan.onclick = () => {
                const input = document.getElementById('total_prc_compute_stock_in_2025');
                if (input) {
                    input.focus();
                    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            };
        }

        // Update diverted compute display
        const divertedComputeSpan = document.getElementById('param-diverted-compute');
        if (divertedComputeSpan) {
            divertedComputeSpan.textContent = formatNumber(divertedCompute);
            // Add click handler
            divertedComputeSpan.onclick = () => {
                const input = document.getElementById('proportion_of_initial_chip_stock_to_divert');
                if (input) {
                    input.focus();
                    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            };
        }

        // Update diversion percentage display
        const diversionPercentSpan = document.getElementById('param-diversion-percent');
        if (diversionPercentSpan) {
            const percentage = (diversionProportion * 100).toFixed(0);
            diversionPercentSpan.textContent = percentage;
            // Add click handler
            diversionPercentSpan.onclick = () => {
                const input = document.getElementById('proportion_of_initial_chip_stock_to_divert');
                if (input) {
                    input.focus();
                    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            };
        }
    }

    // Update initial compute diversion percentage
    if (initialDiversionInput) {
        const percentage = (parseFloat(initialDiversionInput.value) * 100).toFixed(0);
        const spans = [
            document.getElementById('param-initial-diversion-percent'),
            document.getElementById('param-initial-diversion-percent-2')
        ];
        spans.forEach(span => {
            if (span) span.textContent = percentage;
        });
    }

    // Update SME diversion percentage
    const smeDiversionInput = document.getElementById('scanner_proportion');
    if (smeDiversionInput) {
        const percentage = (parseFloat(smeDiversionInput.value) * 100).toFixed(0);
        const spans = [
            document.getElementById('param-sme-diversion-percent'),
            document.getElementById('param-sme-diversion-percent-2')
        ];
        spans.forEach(span => {
            if (span) {
                span.textContent = percentage;
                // Add click handler to focus the input
                span.onclick = () => {
                    const input = document.getElementById('scanner_proportion');
                    if (input) {
                        input.focus();
                        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                };
            }
        });
    }

    // Update fab process node
    const processNodeInput = document.getElementById('process_node');
    if (processNodeInput) {
        const value = processNodeInput.value;
        let nodeText = '28'; // default

        // Parse the process node strategy to extract the nm value
        if (value.includes('28nm') || value === 'nm28') {
            nodeText = '28';
        } else if (value.includes('14nm') || value === 'nm14') {
            nodeText = '14';
        } else if (value.includes('7nm') || value === 'nm7') {
            nodeText = '7';
        } else if (value.includes('130nm') || value === 'nm130') {
            nodeText = '130';
        } else if (value.includes('best_indigenous')) {
            // For "best indigenous" strategies, show the constraint if any
            if (value.includes('gte_28nm')) {
                nodeText = '28';
            } else if (value.includes('gte_14nm')) {
                nodeText = '14';
            } else if (value.includes('gte_7nm')) {
                nodeText = '7';
            } else {
                nodeText = 'best';
            }
        }

        const spans = [
            document.getElementById('param-fab-node'),
            document.getElementById('param-fab-node-2')
        ];
        spans.forEach(span => {
            if (span) {
                span.textContent = nodeText;
                // Add click handler to focus the input
                span.onclick = () => {
                    const input = document.getElementById('process_node');
                    if (input) {
                        input.focus();
                        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                };
            }
        });
    }

    // Update datacenter construction workers
    const datacenterWorkersInput = document.getElementById('datacenter_construction_labor');
    if (datacenterWorkersInput) {
        const workers = parseInt(datacenterWorkersInput.value);
        const formatted = workers.toLocaleString();
        const spans = [
            document.getElementById('param-datacenter-workers'),
            document.getElementById('param-datacenter-workers-2')
        ];
        spans.forEach(span => {
            if (span) span.textContent = formatted;
        });
    }
}