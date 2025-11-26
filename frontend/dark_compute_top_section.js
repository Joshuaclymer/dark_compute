// JavaScript for Dark Compute Model - Top Section (Dashboard and Plots)

// Initialize detection threshold text on page load
document.addEventListener('DOMContentLoaded', function() {
    const thresholdElement = document.getElementById('detection-threshold-text');
    if (thresholdElement && typeof DETECTION_CONFIG !== 'undefined') {
        thresholdElement.textContent = DETECTION_CONFIG.PRIMARY_THRESHOLD;
    }
});

function plotH100YearsTimeSeries(data) {
    if (!data.dark_compute_model) return;

    const dcm = data.dark_compute_model;
    const years = dcm.years;

    // Helper function to format numbers with K, M, B, T suffixes
    function formatNumber(num) {
        if (num >= 1e12) {
            return (num / 1e12).toFixed(1).replace(/\.0$/, '') + 'T';
        } else if (num >= 1e9) {
            return (num / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
        } else if (num >= 1e6) {
            return (num / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
        } else if (num >= 1e3) {
            return (num / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
        }
        return num.toFixed(1);
    }

    // Colors - use blue for probability (same as "Posterior Probability of Covert Project")
    const probColor = '#5B8DBE'; // Blue for probability
    const h100YearsColor = '#5AA89B'; // Turquoise green for H100-years
    const aiRdColor = '#E8A863'; // Orange for AI R&D
    const prcColor = '#C77CAA'; // Purple/pink for PRC compute

    // Get agreement year for calculations
    const agreementYearInput = document.getElementById('agreement_year');
    const agreementYear = agreementYearInput ? parseInt(agreementYearInput.value) : 2030;

    // Calculate cumulative AI R&D H100-years over time using backend data
    // Largest company compute data is now sourced from AI Futures Project input_data.csv
    let aiRdH100Years = [];
    const largestCompanyComputeData = data.initial_stock?.largest_company_compute_over_time;
    if (largestCompanyComputeData && largestCompanyComputeData.length > 0) {
        // Calculate cumulative AI R&D H100-years using trapezoidal integration
        let cumulative = 0;
        aiRdH100Years = [0]; // Start at 0
        for (let i = 0; i < years.length - 1; i++) {
            const yearDuration = years[i + 1] - years[i];
            const avgCompute = (largestCompanyComputeData[i] + largestCompanyComputeData[i + 1]) / 2;
            cumulative += avgCompute * yearDuration;
            aiRdH100Years.push(cumulative);
        }
    }

    // Calculate cumulative PRC AI R&D H100-years over time (if no slowdown)
    const prcCompute2025Input = document.getElementById('total_prc_compute_stock_in_2025');
    const prcGrowthRateInput = document.getElementById('annual_growth_rate_of_prc_compute_stock_p50');
    const prcAiRdFractionInput = document.getElementById('fraction_of_prc_compute_spent_on_ai_rd_before_slowdown');

    let prcH100YearsNoSlowdown = [];
    if (prcCompute2025Input && prcGrowthRateInput && prcAiRdFractionInput) {
        const prcCompute2025 = parseFloat(prcCompute2025Input.value) * 1e6;
        const prcGrowthRate = parseFloat(prcGrowthRateInput.value);
        const prcAiRdFraction = parseFloat(prcAiRdFractionInput.value);

        // Calculate PRC compute for each year (starting from 2025 for projection)
        const prcComputeData = years.map(year => {
            const yearsSince2025 = year - 2025;
            return prcCompute2025 * Math.pow(prcGrowthRate, yearsSince2025);
        });

        // Calculate cumulative PRC AI R&D H100-years using trapezoidal integration starting from agreement year
        let cumulative = 0;
        prcH100YearsNoSlowdown = [];
        for (let i = 0; i < years.length; i++) {
            if (years[i] < agreementYear) {
                prcH100YearsNoSlowdown.push(0); // Before agreement year, set to 0
            } else if (i === 0 || years[i - 1] < agreementYear) {
                prcH100YearsNoSlowdown.push(0); // First year at or after agreement year, start at 0
            } else {
                const yearDuration = years[i] - years[i - 1];
                const avgCompute = (prcComputeData[i] + prcComputeData[i - 1]) / 2;
                cumulative += avgCompute * yearDuration * prcAiRdFraction;
                prcH100YearsNoSlowdown.push(cumulative);
            }
        }
    }

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
            name: 'US Chance of Covert Project (25%-75% CI)',
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
            name: 'Covert H100-Years (25%-75% CI)',
            yaxis: 'y2',
            customdata: dcm.h100_years.median.map(v => formatNumber(v)),
            hovertemplate: 'Covert H100-Years: %{customdata}<extra></extra>'
        },
        // AI R&D H100-years (if calculated)
        ...(aiRdH100Years.length > 0 ? [{
            x: years,
            y: aiRdH100Years,
            type: 'scatter',
            mode: 'lines',
            line: { color: aiRdColor, width: 3, dash: 'dash' },
            name: `Largest AI Company R&D H100-years if no slowdown`,
            yaxis: 'y2',
            customdata: aiRdH100Years.map(v => formatNumber(v)),
            hovertemplate: 'Largest AI Company R&D H100-Years No Slowdown: %{customdata}<extra></extra>'
        }] : []),
        // PRC AI R&D computation if no slowdown (if calculated)
        ...(prcH100YearsNoSlowdown.length > 0 ? [{
            x: years,
            y: prcH100YearsNoSlowdown,
            type: 'scatter',
            mode: 'lines',
            line: { color: prcColor, width: 3, dash: 'dot' },
            name: `PRC AI R&D H100-years if no slowdown`,
            yaxis: 'y2',
            customdata: prcH100YearsNoSlowdown.map(v => formatNumber(v)),
            hovertemplate: 'PRC AI R&D H100-Years No Slowdown: %{customdata}<extra></extra>'
        }] : [])
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true
        },
        yaxis: {
            title: 'US Chance of Covert Project',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            side: 'left',
            range: [0, 1],
            automargin: true
        },
        yaxis2: {
            title: 'H100 years of computation',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            overlaying: 'y',
            side: 'right',
            type: 'log',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.5,
            y: -0.25,
            xanchor: 'center',
            yanchor: 'top',
            orientation: 'h',
            font: { size: 9 },
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
                Plotly.Plots.resize('chipProductionReductionCcdfPlot');
                Plotly.Plots.resize('h100YearsTimeSeriesPlot');
            }, 50);
        }
    }, 150);
}

function plotChipProductionReductionCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.chip_production_reduction_ccdf) {
        document.getElementById('chipProductionReductionCcdfPlot').innerHTML = '<p>No chip production reduction data available</p>';
        return;
    }

    const reductionData = data.dark_compute_model.chip_production_reduction_ccdf;
    const primaryThreshold = DETECTION_CONFIG.PRIMARY_THRESHOLD;

    // Check if we have the new structure (with global and prc keys)
    const globalCcdf = reductionData.global ? reductionData.global[primaryThreshold] : null;
    const prcCcdf = reductionData.prc ? reductionData.prc[primaryThreshold] : null;

    if ((!globalCcdf || globalCcdf.length === 0) && (!prcCcdf || prcCcdf.length === 0)) {
        document.getElementById('chipProductionReductionCcdfPlot').innerHTML = `<p>No chip production reduction data available for ${primaryThreshold}x update</p>`;
        return;
    }

    const traces = [];

    // Add global AI chip production reduction trace
    if (globalCcdf && globalCcdf.length > 0) {
        traces.push({
            x: globalCcdf.map(d => d.x),
            y: globalCcdf.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#5B8DBE', width: 2 },
            name: 'Global Production',
            hovertemplate: 'Reduction: %{x:.1f}x<br>P(≥x): %{y:.3f}<extra></extra>'
        });
    }

    // Add PRC AI chip production reduction trace
    if (prcCcdf && prcCcdf.length > 0) {
        traces.push({
            x: prcCcdf.map(d => d.x),
            y: prcCcdf.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#C77CAA', width: 2 },
            name: 'PRC Production',
            hovertemplate: 'Reduction: %{x:.1f}x<br>P(≥x): %{y:.3f}<extra></extra>'
        });
    }

    const layout = {
        xaxis: {
            title: "No-agreement production<br>/ covert production",
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            type: 'log',
            range: [Math.log10(1), null],  // Start at 1x reduction
            automargin: true,
            ticksuffix: 'x'
        },
        yaxis: {
            title: 'P(Reduction > x)',
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

    Plotly.newPlot('chipProductionReductionCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('chipProductionReductionCcdfPlot'), 50);
}

function plotAiRdReductionCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.ai_rd_reduction_ccdf) {
        document.getElementById('aiRdReductionCcdfPlot').innerHTML = '<p>No AI R&D reduction data available</p>';
        return;
    }

    const reductionData = data.dark_compute_model.ai_rd_reduction_ccdf;
    const primaryThreshold = DETECTION_CONFIG.PRIMARY_THRESHOLD;

    // Check if we have the new structure (with largest_company and prc keys)
    const largestCompanyCcdf = reductionData.largest_company ? reductionData.largest_company[primaryThreshold] : reductionData[primaryThreshold];
    const prcCcdf = reductionData.prc ? reductionData.prc[primaryThreshold] : null;

    if (!largestCompanyCcdf || largestCompanyCcdf.length === 0) {
        document.getElementById('aiRdReductionCcdfPlot').innerHTML = `<p>No AI R&D reduction data available for ${primaryThreshold}x update</p>`;
        return;
    }

    const traces = [];

    // Add largest company AI R&D reduction trace
    traces.push({
        x: largestCompanyCcdf.map(d => d.x),
        y: largestCompanyCcdf.map(d => d.y),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#E8A863', width: 2 },
        name: 'Largest AI Company',
        hovertemplate: 'Reduction: %{x:.1f}x<br>P(≥x): %{y:.3f}<extra></extra>'
    });

    // Add PRC AI R&D reduction trace if available
    if (prcCcdf && prcCcdf.length > 0) {
        traces.push({
            x: prcCcdf.map(d => d.x),
            y: prcCcdf.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#C77CAA', width: 2 },
            name: 'PRC Compute',
            hovertemplate: 'Reduction: %{x:.1f}x<br>P(≥x): %{y:.3f}<extra></extra>'
        });
    }

    const layout = {
        xaxis: {
            title: "No-agreement computation<br>/ covert computation",
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            type: 'log',
            range: [Math.log10(10), null],  // Start at 10x reduction
            automargin: true,
            ticksuffix: 'x'
        },
        yaxis: {
            title: 'P(Reduction > x)',
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

    Plotly.newPlot('aiRdReductionCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('aiRdReductionCcdfPlot'), 50);
}

function plotProjectH100YearsCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.h100_years_ccdf) {
        document.getElementById('projectH100YearsCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    // Use likelihood ratios from backend or global config
    const likelihoodRatios = data.dark_compute_model.likelihood_ratios || DETECTION_CONFIG.LIKELIHOOD_RATIOS;
    const colors = DETECTION_CONFIG.COLORS;

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
            title: "H100-years of covert computation",
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

function plotTimeToDetectionCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.time_to_detection_ccdf) {
        document.getElementById('timeToDetectionCcdfPlot').innerHTML = '<p>No time to detection data available</p>';
        return;
    }

    // Use likelihood ratios from backend or global config
    const likelihoodRatios = data.dark_compute_model.likelihood_ratios || DETECTION_CONFIG.LIKELIHOOD_RATIOS;
    const colors = DETECTION_CONFIG.COLORS;

    const traces = [];
    const thresholds = likelihoodRatios.map((lr, index) => {
        return {
            value: lr,
            label: `>${lr}x update`,
            color: colors[index % colors.length]
        };
    });

    // Reverse thresholds for legend order (highest to lowest)
    const thresholdsReversed = [...thresholds].reverse();

    for (const threshold of thresholdsReversed) {
        const ccdf = data.dark_compute_model.time_to_detection_ccdf[threshold.value];
        if (ccdf && ccdf.length > 0) {
            traces.push({
                x: ccdf.map(d => d.x),
                y: ccdf.map(d => d.y),
                type: 'scatter',
                mode: 'lines',
                line: { color: threshold.color, width: 2 },
                name: `"Detection" = ${threshold.label}`,
                hovertemplate: 'Years: %{x:.1f}<br>P(≥x): %{y:.3f}<extra></extra>'
            });
        }
    }

    const layout = {
        xaxis: {
            title: "Years until detection",
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            automargin: true
        },
        yaxis: {
            title: 'P(Time to detection > x)',
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

    Plotly.newPlot('timeToDetectionCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('timeToDetectionCcdfPlot'), 50);
}

function roundToSigFigs(num, sigFigs) {
    // Round a number to specified significant figures
    if (num === 0) return 0;
    const magnitude = Math.floor(Math.log10(Math.abs(num)));
    const multiplier = Math.pow(10, sigFigs - magnitude - 1);
    return Math.round(num * multiplier) / multiplier;
}

function getMedianFromCcdf(ccdfData) {
    // Find the x value where y ≈ 0.5 (median) from a CCDF
    // CCDF is sorted by x ascending, y descending (starts at ~1, ends at ~0)
    if (!ccdfData || ccdfData.length === 0) return null;

    // Find the point where y crosses 0.5
    for (let i = 0; i < ccdfData.length - 1; i++) {
        if (ccdfData[i].y >= 0.5 && ccdfData[i + 1].y < 0.5) {
            // Linear interpolation to find exact x where y = 0.5
            const x1 = ccdfData[i].x;
            const x2 = ccdfData[i + 1].x;
            const y1 = ccdfData[i].y;
            const y2 = ccdfData[i + 1].y;
            const t = (0.5 - y1) / (y2 - y1);
            return x1 + t * (x2 - x1);
        }
    }

    // If y never crosses 0.5, return the last x value (or first if all > 0.5)
    if (ccdfData[0].y < 0.5) return ccdfData[0].x;
    return ccdfData[ccdfData.length - 1].x;
}

function updateDarkComputeModelDashboard(data) {
    // Update the dark compute model dashboard with median project metrics

    const projectH100YearsValues = data.dark_compute_model?.individual_project_h100_years_before_detection || [];
    const projectTimeValues = data.dark_compute_model?.individual_project_time_before_detection || [];

    if (projectH100YearsValues.length > 0 && projectTimeValues.length > 0) {
        // Calculate medians
        const sortedH100Years = [...projectH100YearsValues].sort((a, b) => a - b);
        const sortedTime = [...projectTimeValues].sort((a, b) => a - b);

        const medianIdx = Math.floor(sortedH100Years.length / 2);
        const projectH100YearsMedian = sortedH100Years[medianIdx];
        const projectTimeMedian = sortedTime[medianIdx];

        // Display H100-years (covert computation)
        const h100YearsRounded = Math.round(projectH100YearsMedian / 100000) * 100000;
        if (h100YearsRounded >= 1000000) {
            document.getElementById('dashboardMedianH100Years').textContent =
                `${(h100YearsRounded / 1000000).toFixed(1)}M H100-years`;
        } else if (h100YearsRounded >= 1000) {
            document.getElementById('dashboardMedianH100Years').textContent =
                `${(h100YearsRounded / 1000).toFixed(0)}K H100-years`;
        } else {
            document.getElementById('dashboardMedianH100Years').textContent =
                `${h100YearsRounded.toFixed(0)} H100-years`;
        }

        // Display time to detection
        document.getElementById('dashboardMedianTime').textContent = projectTimeMedian.toFixed(1) + ' years';

        // Calculate AI R&D reduction (updates dashboardAiRdReduction)
        updateAiRdReduction(data);
    }

    // Update chip production reduction from the CCDF data
    updateChipProductionReduction(data);
}

function updateChipProductionReduction(data) {
    // Get the median from the global chip production reduction CCDF
    const primaryThreshold = DETECTION_CONFIG.PRIMARY_THRESHOLD;
    const reductionData = data.dark_compute_model?.chip_production_reduction_ccdf;

    if (reductionData && reductionData.global && reductionData.global[primaryThreshold]) {
        const globalCcdf = reductionData.global[primaryThreshold];
        const medianReduction = getMedianFromCcdf(globalCcdf);

        if (medianReduction !== null) {
            const rounded = roundToSigFigs(medianReduction, 1);
            document.getElementById('dashboardPrcReduction').textContent = rounded + 'x';
        } else {
            document.getElementById('dashboardPrcReduction').textContent = '--';
        }
    } else {
        document.getElementById('dashboardPrcReduction').textContent = '--';
    }
}

function updateAiRdReduction(data) {
  /**
   * Get the median AI R&D reduction ratio from the backend CCDF data.
   * This uses the same data as the "Reduction in AI R&D computation during agreement" plot.
   */

  try {
    const primaryThreshold = DETECTION_CONFIG.PRIMARY_THRESHOLD;
    const reductionData = data.dark_compute_model?.ai_rd_reduction_ccdf;

    if (reductionData && reductionData.largest_company && reductionData.largest_company[primaryThreshold]) {
      const largestCompanyCcdf = reductionData.largest_company[primaryThreshold];
      const medianReduction = getMedianFromCcdf(largestCompanyCcdf);

      if (medianReduction !== null) {
        const rounded = roundToSigFigs(medianReduction, 1);
        document.getElementById('dashboardAiRdReduction').textContent = rounded + 'x';
      } else {
        document.getElementById('dashboardAiRdReduction').textContent = '--';
      }
    } else {
      document.getElementById('dashboardAiRdReduction').textContent = '--';
    }

  } catch (error) {
    console.error('AI R&D Reduction Error:', error.message);
    document.getElementById('dashboardAiRdReduction').textContent = '--';
  }
}
