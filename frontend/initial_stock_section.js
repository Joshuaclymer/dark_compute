// JavaScript for Initial Stock section

function plotInitialStock(data) {
    // Plot detection probability bar chart and initial compute stock histogram
    if (data.initial_stock && data.initial_stock.initial_dark_compute_detection_probs && data.initial_stock.initial_compute_stock_samples) {

        // Update dashboard with median values
        const sortedDarkCompute = [...data.initial_stock.initial_compute_stock_samples].sort((a, b) => a - b);
        const medianDarkCompute = sortedDarkCompute[Math.floor(sortedDarkCompute.length / 2)];

        // Format H100e text
        let h100eText;
        const rounded = Math.round(medianDarkCompute / 100000) * 100000;
        if (rounded >= 1000000) {
            h100eText = `${(rounded / 1000000).toFixed(1)}M H100e`;
        } else if (rounded >= 1000) {
            h100eText = `${(rounded / 1000).toFixed(0)}K H100e`;
        } else {
            h100eText = `${rounded.toFixed(0)} H100e`;
        }

        // Calculate energy for this compute stock
        const h100PowerWatts = parseFloat(document.getElementById('h100_power_watts').value);
        const energyGW = (medianDarkCompute * h100PowerWatts) / 1e9; // Convert watts to GW

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
                titlefont: { size: 11 },
                tickfont: { size: 9 }
            },
            yaxis: {
                title: 'P(Detection)',
                titlefont: { size: 11 },
                tickfont: { size: 10 },
                range: [0, 1],
                tickformat: '.0%'
            },
            showlegend: false,
            margin: { l: 55, r: 10, t: 10, b: 70 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot('initialDarkComputeDetectionPlot', [barTrace], barLayout, {displayModeBar: false, responsive: true});

        // Histogram for initial compute stock
        plotPDF('initialComputeStockPlot', data.initial_stock.initial_compute_stock_samples, '#9B72B0', 'PRC Dark Compute Stock (H100e)', 30, false);
    }

    // Plot LR breakdown histograms for initial compute reporting
    if (data.initial_stock && data.initial_stock.lr_prc_accounting_samples && data.initial_stock.lr_global_accounting_samples && data.initial_stock.lr_combined_samples) {
        // Use log scale with range 1/3 to 5, and blue color #5B8DBE
        plotPDF('lrPrcAccountingPlot', data.initial_stock.lr_prc_accounting_samples, '#5B8DBE', 'Likelihood Ratio from PRC Accounting', 12, true, 1/3, 5);
        plotPDF('lrGlobalAccountingPlot', data.initial_stock.lr_global_accounting_samples, '#5B8DBE', 'Likelihood Ratio from Global Production Accounting', 12, true, 1/3, 5);
        plotPDF('lrCombinedPlot', data.initial_stock.lr_combined_samples, '#5B8DBE', 'Combined LR', 12, true, 1/3, 5);
    }

    // Plot initial dark compute stock breakdown
    if (data.initial_stock && data.initial_stock.initial_prc_stock_samples && data.initial_stock.diversion_proportion && data.initial_stock.initial_compute_stock_samples) {

        // Plot initial PRC stock distribution - purple color #9B72B0
        plotPDF('initialPrcStockPlot', data.initial_stock.initial_prc_stock_samples, '#9B72B0', 'Initial PRC Compute Stock (H100e)', 30, false);

        // Display the diversion proportion
        const diversionPercent = (data.initial_stock.diversion_proportion * 100).toFixed(0);
        document.getElementById('diversionProportionDisplay').innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                <div class="breakdown-box-inner">${diversionPercent}%</div>
                <div class="breakdown-label">Proportion diverted<br>to covert project</div>
            </div>`;

        // Plot the resulting dark compute stock - purple color #9B72B0
        plotPDF('darkComputeResultPlot', data.initial_stock.initial_compute_stock_samples, '#9B72B0', 'Dark Compute Stock (H100e)', 30, false);
    }
}
