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

    // Calculate cumulative AI R&D H100-years over time
    const globalCompute2026Input = document.getElementById('median_global_compute_in_2026');
    const growthRateInput = document.getElementById('median_global_compute_annual_rate_of_increase');
    const fractionInput = document.getElementById('fraction_of_global_compute_for_single_ai_project');

    let aiRdH100Years = [];
    if (globalCompute2026Input && growthRateInput && fractionInput) {
        const globalCompute2026 = parseFloat(globalCompute2026Input.value) * 1e6;
        const growthRate = parseFloat(growthRateInput.value);
        const fractionOfGlobalCompute = parseFloat(fractionInput.value);

        // Calculate global compute for each year
        const globalComputeData = years.map(year => {
            const yearsSince2026 = year - 2026;
            return globalCompute2026 * Math.pow(growthRate, yearsSince2026);
        });

        // Calculate cumulative AI R&D H100-years using trapezoidal integration
        let cumulative = 0;
        aiRdH100Years = [0]; // Start at 0
        for (let i = 0; i < years.length - 1; i++) {
            const yearDuration = years[i + 1] - years[i];
            const avgCompute = (globalComputeData[i] + globalComputeData[i + 1]) / 2;
            cumulative += avgCompute * yearDuration * fractionOfGlobalCompute;
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
            name: `Global AI R&D H100-years if no slowdown`,
            yaxis: 'y2',
            customdata: aiRdH100Years.map(v => formatNumber(v)),
            hovertemplate: 'Largest Project AI R&D H100-Years No Slowdown: %{customdata}<extra></extra>'
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
                Plotly.Plots.resize('projectH100YearsCcdfPlot');
                Plotly.Plots.resize('h100YearsTimeSeriesPlot');
            }, 50);
        }
    }, 150);
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

function plotAiRdReductionCcdf(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.ai_rd_reduction_ccdf) {
        document.getElementById('aiRdReductionCcdfPlot').innerHTML = '<p>No AI R&D reduction data available</p>';
        return;
    }

    const reductionData = data.dark_compute_model.ai_rd_reduction_ccdf;
    const primaryThreshold = DETECTION_CONFIG.PRIMARY_THRESHOLD;

    // Check if we have the new structure (with global and prc keys)
    const globalCcdf = reductionData.global ? reductionData.global[primaryThreshold] : reductionData[primaryThreshold];
    const prcCcdf = reductionData.prc ? reductionData.prc[primaryThreshold] : null;

    if (!globalCcdf || globalCcdf.length === 0) {
        document.getElementById('aiRdReductionCcdfPlot').innerHTML = `<p>No AI R&D reduction data available for ${primaryThreshold}x update</p>`;
        return;
    }

    const traces = [];

    // Add global AI R&D reduction trace
    traces.push({
        x: globalCcdf.map(d => d.x),
        y: globalCcdf.map(d => d.y),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#E8A863', width: 2 },
        name: 'Global AI R&D Compute',
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
            name: 'PRC AI R&D Compute',
            hovertemplate: 'Reduction: %{x:.1f}x<br>P(≥x): %{y:.3f}<extra></extra>'
        });
    }

    const layout = {
        xaxis: {
            title: "Reduction in AI R&D Compute",
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            type: 'log',
            range: [Math.log10(10), null],  // Start at 10x reduction
            automargin: true
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

        // Calculate AI R&D reduction (will update both dashboard values)
        updateAiRdReduction(data, projectTimeMedian, projectH100YearsMedian);
    }
}

function updateAiRdReduction(data, projectTimeMedian, projectH100YearsMedian) {
  /**
   * Calculate the ratio of average global operating AI chips to average covert project operating AI chips.
   *
   * This function:
   * 1. Calculates the average number of operating AI chips (H100 equivalents) for the covert project
   *    between the agreement year and detection year
   * 2. Calculates the average number of AI chips operating globally (H100 equivalents)
   *    in the same time period (using the fraction parameter for a single large AI project)
   * 3. Returns the ratio: (global average operating AI chips) / (covert project average operating AI chips)
   *
   * This ratio represents how many times larger the global AI R&D compute is compared to the covert project.
   */

  try {
    console.log('AI R&D Calculation Debug:');
    console.log('data:', data);
    console.log('projectTimeMedian:', projectTimeMedian);

    // Check if data exists
    if (!data.dark_compute_model) {
      throw new Error('data.dark_compute_model is undefined or null');
    }

    // Get the operational dark compute data (operating AI chips for covert project)
    const operationalDarkCompute = data.dark_compute_model.operational_dark_compute;
    if (!operationalDarkCompute || !operationalDarkCompute.median) {
      throw new Error('operational_dark_compute data not found');
    }

    const years = data.dark_compute_model.years;
    if (!years) {
      throw new Error('years data not found');
    }

    // Get agreement year from input
    const agreementYearInput = document.getElementById('agreement_year');
    if (!agreementYearInput) {
      throw new Error('Input element with id "agreement_year" not found');
    }
    const agreementYear = parseInt(agreementYearInput.value);

    // Calculate detection year
    const detectionYear = agreementYear + projectTimeMedian;

    // Find the indices for agreement year and detection year
    const agreementIdx = years.findIndex(y => y >= agreementYear);
    const detectionIdx = years.findIndex(y => y >= detectionYear);

    console.log('agreementYear:', agreementYear);
    console.log('detectionYear:', detectionYear);
    console.log('agreementIdx:', agreementIdx);
    console.log('detectionIdx:', detectionIdx);

    if (agreementIdx === -1) {
      throw new Error(`Agreement year ${agreementYear} not found in years array`);
    }

    if (detectionIdx === -1) {
      throw new Error(`Detection year ${detectionYear} not found in years array`);
    }

    // Calculate average operating AI chips for covert project between agreement and detection
    let covertSum = 0;
    let numPoints = 0;
    for (let i = agreementIdx; i <= detectionIdx; i++) {
      covertSum += operationalDarkCompute.median[i];
      numPoints++;
    }
    const covertAverage = covertSum / numPoints;

    // Get global compute parameters
    const globalCompute2026Input = document.getElementById('median_global_compute_in_2026');
    const growthRateInput = document.getElementById('median_global_compute_annual_rate_of_increase');
    const fractionInput = document.getElementById('fraction_of_global_compute_for_single_ai_project');

    if (!globalCompute2026Input || !growthRateInput || !fractionInput) {
      throw new Error('Global compute parameter inputs not found');
    }

    const globalCompute2026 = parseFloat(globalCompute2026Input.value) * 1e6;
    const growthRate = parseFloat(growthRateInput.value);
    const fractionOfGlobalCompute = parseFloat(fractionInput.value);

    // Calculate average global operating AI chips between agreement and detection
    let globalSum = 0;
    for (let i = agreementIdx; i <= detectionIdx; i++) {
      const year = years[i];
      const yearsSince2026 = year - 2026;
      const globalCompute = globalCompute2026 * Math.pow(growthRate, yearsSince2026);
      // Apply fraction for single large AI project
      globalSum += globalCompute * fractionOfGlobalCompute;
    }
    const globalAverage = globalSum / numPoints;

    // Calculate the ratio: global / covert
    const ratio = globalAverage / covertAverage;

    console.log('covertAverage:', covertAverage);
    console.log('globalAverage:', globalAverage);
    console.log('ratio:', ratio);

    // Update dashboard display for AI R&D reduction
    document.getElementById('dashboardAiRdReduction').textContent = ratio.toFixed(1) + 'x';

    // Now calculate PRC AI R&D compute reduction
    // Get PRC compute parameters
    const prcCompute2025Input = document.getElementById('total_prc_compute_stock_in_2025');
    const prcGrowthRateInput = document.getElementById('annual_growth_rate_of_prc_compute_stock_p50');
    const prcAiRdFractionInput = document.getElementById('fraction_of_prc_compute_spent_on_ai_rd_before_slowdown');

    if (prcCompute2025Input && prcGrowthRateInput && prcAiRdFractionInput) {
      const prcCompute2025 = parseFloat(prcCompute2025Input.value) * 1e6;
      const prcGrowthRate = parseFloat(prcGrowthRateInput.value);
      const prcAiRdFraction = parseFloat(prcAiRdFractionInput.value);

      // Calculate average PRC AI R&D compute if no slowdown (continues growing)
      // This is total PRC compute * fraction spent on AI R&D
      let prcAiRdNoSlowdownSum = 0;
      for (let i = agreementIdx; i <= detectionIdx; i++) {
        const year = years[i];
        const yearsSince2025 = year - 2025;
        const prcCompute = prcCompute2025 * Math.pow(prcGrowthRate, yearsSince2025);
        // Apply the fraction spent on AI R&D
        prcAiRdNoSlowdownSum += prcCompute * prcAiRdFraction;
      }
      const prcAiRdNoSlowdownAverage = prcAiRdNoSlowdownSum / numPoints;

      // The average with slowdown is the covert project's operating compute (same as calculated above)
      const prcAiRdWithSlowdownAverage = covertAverage;

      // Calculate the ratio: no slowdown / with slowdown
      const prcRatio = prcAiRdNoSlowdownAverage / prcAiRdWithSlowdownAverage;

      console.log('prcAiRdNoSlowdownAverage:', prcAiRdNoSlowdownAverage);
      console.log('prcAiRdWithSlowdownAverage:', prcAiRdWithSlowdownAverage);
      console.log('prcAiRdFraction:', prcAiRdFraction);
      console.log('prcRatio:', prcRatio);

      // Update dashboard display for PRC AI R&D reduction
      document.getElementById('dashboardPrcReduction').textContent = prcRatio.toFixed(1) + 'x';
    } else {
      document.getElementById('dashboardPrcReduction').textContent = '--';
    }

  } catch (error) {
    console.error('AI R&D Calculation Error:', error.message);
    console.error('Full error:', error);
    document.getElementById('dashboardAiRdReduction').textContent = '--';
    document.getElementById('dashboardPrcReduction').textContent = '--';
    // Re-throw the error so it appears in the browser console
    throw error;
  }
}
