// JavaScript for Initial Stock section

function plotInitialStock(data) {
    // Plot PRC compute over time
    if (data.initial_stock && data.initial_stock.prc_compute_over_time && data.initial_stock.prc_compute_years) {
        const years = data.initial_stock.prc_compute_years;
        const p25 = data.initial_stock.prc_compute_over_time.p25;
        const median = data.initial_stock.prc_compute_over_time.median;
        const p75 = data.initial_stock.prc_compute_over_time.p75;
        const domesticMedian = data.initial_stock.prc_domestic_compute_over_time.median;
        const proportionDomestic = data.initial_stock.proportion_domestic;

        // Create shaded area for 25-75 percentile range
        const shadedArea = {
            x: [...years, ...years.slice().reverse()],
            y: [...p75, ...p25.slice().reverse()],
            fill: 'toself',
            fillcolor: 'rgba(155, 114, 176, 0.2)',
            line: { color: 'transparent' },
            type: 'scatter',
            showlegend: false,
            hoverinfo: 'skip',
            name: '25th-75th percentile'
        };

        // Create 25th percentile line
        const p25Line = {
            x: years,
            y: p25,
            mode: 'lines',
            line: { color: '#9B72B0', width: 1, dash: 'dash' },
            type: 'scatter',
            showlegend: false,
            hovertemplate: 'Year: %{x}<br>25th percentile: %{y:.2s} H100e<extra></extra>'
        };

        // Create median line
        const medianLine = {
            x: years,
            y: median,
            mode: 'lines',
            line: { color: '#9B72B0', width: 2 },
            type: 'scatter',
            name: 'Median',
            hovertemplate: 'Year: %{x}<br>Median: %{y:.2s} H100e<extra></extra>'
        };

        // Create 75th percentile line
        const p75Line = {
            x: years,
            y: p75,
            mode: 'lines',
            line: { color: '#9B72B0', width: 1, dash: 'dash' },
            type: 'scatter',
            showlegend: false,
            hovertemplate: 'Year: %{x}<br>75th percentile: %{y:.2s} H100e<extra></extra>'
        };

        // Create dummy trace for percentile range legend entry
        const percentileRangeLegend = {
            x: [years[0], years[1]],
            y: [null, null],
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(155, 114, 176, 0.2)',
            line: { color: 'transparent' },
            name: '25th-75th %tile',
            showlegend: true,
            hoverinfo: 'skip'
        };

        // Create domestic production line
        const domesticLine = {
            x: years,
            y: domesticMedian,
            mode: 'lines',
            line: { color: '#5AA89B', width: 2, dash: 'dot' },
            type: 'scatter',
            name: 'Domestically produced',
            hovertemplate: 'Year: %{x}<br>Domestically produced: %{y:.2s} H100e<br>(' + (proportionDomestic * 100).toFixed(0) + '% of total)<extra></extra>'
        };

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true
            },
            yaxis: {
                title: 'PRC chip stock (H100e)',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                type: 'log',
                automargin: true
            },
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: '#ccc',
                borderwidth: 1,
                font: { size: 7 }
            },
            margin: { l: 50, r: 20, t: 10, b: 55, pad: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true
        };

        Plotly.newPlot('prcComputeOverTimePlot', [shadedArea, p25Line, medianLine, p75Line, percentileRangeLegend, domesticLine], layout, {displayModeBar: false, responsive: true});

        // Update agreement year text
        const agreementYear = years[years.length - 1];
        const agreementYearElement = document.getElementById('agreementYearValue');
        if (agreementYearElement) {
            agreementYearElement.textContent = agreementYear;
        }

        // Update parameter displays in caption
        // PRC compute stock in 2025
        const prcCompute2025Span = document.getElementById('param-prc-compute-2025');
        if (prcCompute2025Span) {
            const prcCompute2025Input = document.getElementById('total_prc_compute_stock_in_2025');
            if (prcCompute2025Input) {
                const value = parseFloat(prcCompute2025Input.value);
                prcCompute2025Span.textContent = value.toFixed(1);
                // Add click handler
                prcCompute2025Span.onclick = () => {
                    prcCompute2025Input.focus();
                    prcCompute2025Input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                };
            }
        }

        // Growth rate median (display as multiplier)
        const growthRateMedianSpan = document.getElementById('param-growth-rate-median');
        if (growthRateMedianSpan) {
            const growthRateMedianInput = document.getElementById('annual_growth_rate_of_prc_compute_stock_p50');
            if (growthRateMedianInput) {
                const value = parseFloat(growthRateMedianInput.value);
                growthRateMedianSpan.textContent = value.toFixed(1);
                // Add click handler
                growthRateMedianSpan.onclick = () => {
                    growthRateMedianInput.focus();
                    growthRateMedianInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                };
            }
        }
    }

    // Plot detection probability bar chart and initial compute stock histogram
    if (data.initial_stock && data.initial_stock.initial_dark_compute_detection_probs && data.initial_stock.initial_compute_stock_samples) {

        // Update dashboard with 80th percentile values
        const sortedDarkCompute = [...data.initial_stock.initial_compute_stock_samples].sort((a, b) => a - b);
        const p80DarkCompute = sortedDarkCompute[Math.floor(sortedDarkCompute.length * 0.8)];

        // Format H100e text
        let h100eText;
        const rounded = Math.round(p80DarkCompute / 100000) * 100000;
        if (rounded >= 1000000) {
            h100eText = `${(rounded / 1000000).toFixed(1)}M H100e`;
        } else if (rounded >= 1000) {
            h100eText = `${(rounded / 1000).toFixed(0)}K H100e`;
        } else {
            h100eText = `${rounded.toFixed(0)} H100e`;
        }

        // Calculate energy for this compute stock
        const h100PowerWatts = parseFloat(document.getElementById('h100_power_watts').value);
        const energyGW = (p80DarkCompute * h100PowerWatts) / 1e9; // Convert watts to GW

        let energyText;
        if (energyGW >= 1) {
            energyText = `${energyGW.toFixed(1)} GW`;
        } else if (energyGW >= 0.001) {
            energyText = `${(energyGW * 1000).toFixed(0)} MW`;
        } else {
            energyText = `${(energyGW * 1000).toFixed(1)} MW`;
        }

        document.getElementById('dashboardMedianDarkComputeCombined').innerHTML =
            `${h100eText}<br><span style="font-size: 24px; color: #666;">${energyText}</span>`;

        // Display probability of detection (P(LR ≥ 5x))
        if (data.initial_stock.initial_dark_compute_detection_probs && data.initial_stock.initial_dark_compute_detection_probs['5x'] !== undefined) {
            const probDetection = data.initial_stock.initial_dark_compute_detection_probs['5x'];
            document.getElementById('dashboardMedianLR').textContent = (probDetection * 100).toFixed(1) + '%';
        }

        // Bar chart for detection probabilities
        const likelihoodRatios = data.likelihood_ratios || [1, 3, 5];
        const colors = ['#9B7BB3', '#5B8DBE', '#5AA89B'];  // Purple, Blue, Blue-green
        const detectionProbs = likelihoodRatios.map((lr, idx) => ({
            lr: lr,
            prob: data.initial_stock.initial_dark_compute_detection_probs[`${lr}x`] || 0,
            color: colors[idx]
        }));

        const barTrace = {
            x: detectionProbs.map(d => `Detection means<br>≥${d.lr}x LR`),
            y: detectionProbs.map(d => d.prob),
            type: 'bar',
            marker: {
                color: detectionProbs.map(d => d.color)
            },
            hovertemplate: 'P(LR ≥ %{x}): %{y:.2%}<extra></extra>'
        };

        const barLayout = {
            xaxis: {
                title: 'Probability of detection',
                titlefont: { size: 13 },
                tickfont: { size: 10 },
                automargin: true
            },
            yaxis: {
                title: {
                    text: 'P(Detection)',
                    standoff: 15
                },
                titlefont: { size: 13 },
                tickfont: { size: 10 },
                range: [0, 1],
                tickformat: '.0%',
                automargin: true
            },
            showlegend: false,
            margin: { l: 55, r: 0, t: 0, b: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true
        };

        Plotly.newPlot('initialDarkComputeDetectionPlot', [barTrace], barLayout, {displayModeBar: false, responsive: true});

        // Histogram for initial compute stock
        plotPDF('initialComputeStockPlot', data.initial_stock.initial_compute_stock_samples, '#9B72B0', 'PRC Dark Compute Stock (H100 equivalents (FLOPS))', 30, false, null, null, null, null, 'log');

        // Match plot heights to dashboard height after both plots are created
        setTimeout(() => {
            const dashboard = document.querySelector('#initialStockTopSection .dashboard');
            const plotContainers = document.querySelectorAll('#initialStockTopSection .plot-container');
            if (dashboard && plotContainers.length > 0) {
                const dashboardHeight = dashboard.offsetHeight;
                plotContainers.forEach(container => {
                    container.style.height = dashboardHeight + 'px';
                });
                // Force resize after setting height
                setTimeout(() => {
                    Plotly.Plots.resize('initialDarkComputeDetectionPlot');
                    Plotly.Plots.resize('initialComputeStockPlot');
                }, 50);
            }
        }, 150);
    }

    // Plot LR breakdown histograms for initial compute reporting
    if (data.initial_stock && data.initial_stock.lr_prc_accounting_samples) {
        // Use log scale with range 1/3 to 5, and blue color #5B8DBE
        plotPDF('lrPrcAccountingPlot', data.initial_stock.lr_prc_accounting_samples, '#5B8DBE', 'Likelihood Ratio from PRC Accounting', 12, true, 1/3, 5);
    }

    // Plot initial dark compute stock breakdown
    if (data.initial_stock && data.initial_stock.initial_prc_stock_samples && data.initial_stock.diversion_proportion && data.initial_stock.initial_compute_stock_samples) {

        // Plot initial PRC stock distribution - purple color #9B72B0
        plotPDF('initialPrcStockPlot', data.initial_stock.initial_prc_stock_samples, '#9B72B0', 'Initial PRC Compute Stock (H100 equivalents (FLOPS))', 30, false, null, null, null, null, 'log');

        // Display the diversion proportion
        const diversionPercent = (data.initial_stock.diversion_proportion * 100).toFixed(0);
        document.getElementById('diversionProportionDisplay').innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                <div class="breakdown-box-inner">${diversionPercent}%</div>
                <div class="breakdown-label">Proportion diverted<br>to covert project</div>
            </div>`;

        // Attach hover effect to the newly created breakdown-box-inner
        const diversionProportionInner = document.querySelector('#diversionProportionDisplay .breakdown-box-inner');
        if (diversionProportionInner) {
            diversionProportionInner.style.transition = 'all 0.2s ease';
            diversionProportionInner.addEventListener('mouseenter', () => {
                diversionProportionInner.style.boxShadow = '0 0 6px rgba(0, 123, 255, 0.25)';
                diversionProportionInner.style.transform = 'scale(1.015)';
            });
            diversionProportionInner.addEventListener('mouseleave', () => {
                diversionProportionInner.style.boxShadow = '';
                diversionProportionInner.style.transform = '';
            });
        }

        // Plot the resulting dark compute stock - purple color #9B72B0
        plotPDF('darkComputeResultPlot', data.initial_stock.initial_compute_stock_samples, '#9B72B0', 'Dark Compute Stock (H100 equivalents (FLOPS))', 30, false, null, null, null, null, 'log');
    }
}
