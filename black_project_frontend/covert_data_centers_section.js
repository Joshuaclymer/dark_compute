// JavaScript for Covert Datacenters section

function plotDatacenterCombined(data) {
    if (!data.black_project_model || !data.black_project_model.datacenter_capacity) return;
    if (!data.black_datacenters || !data.black_datacenters.lr_datacenters) return;

    const dcm = data.black_project_model;
    const cd = data.black_datacenters;
    const years = dcm.years;

    // Colors
    const capacityColor = '#5AA89B'; // Turquoise for capacity
    const lrColor = '#5B8DBE'; // Blue for likelihood ratio

    // Create traces for both series
    const traces = [
        // LR: 25th percentile (invisible)
        {
            x: years,
            y: cd.lr_datacenters.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // LR: 75th percentile (fills to previous)
        {
            x: years,
            y: cd.lr_datacenters.p75,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: lrColor + '30',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // LR: Median line
        {
            x: years,
            y: cd.lr_datacenters.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: lrColor, width: 3 },
            name: 'Evidence of Datacenters',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>',
            yaxis: 'y'
        },
        // Capacity: Upper bound (invisible)
        {
            x: years,
            y: dcm.datacenter_capacity.p75,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
        },
        // Capacity: Lower bound (fills to previous)
        {
            x: years,
            y: dcm.datacenter_capacity.p25,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(90, 168, 155, 0.2)',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
        },
        // Capacity: Median line
        {
            x: years,
            y: dcm.datacenter_capacity.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: capacityColor, width: 3 },
            name: 'Covert Datacenter capacity',
            hovertemplate: 'Capacity: %{y:.2f} GW<extra></extra>',
            yaxis: 'y2'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true
        },
        yaxis: {
            title: 'Evidence (LR)',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            automargin: true,
            side: 'left',
            type: 'log'
        },
        yaxis2: {
            title: 'Capacity (GW)',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            overlaying: 'y',
            side: 'right',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.98,
            y: 0.02,
            xanchor: 'right',
            yanchor: 'bottom',
            orientation: 'v',
            font: { size: 10 },
            bgcolor: 'rgba(255,255,255,0.8)',
            borderwidth: 1,
            bordercolor: '#ddd'
        },
        hovermode: 'x unified',
        margin: { l: 55, r: 55, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('datacenterCombinedPlot', traces, layout, {displayModeBar: false, responsive: true});

    // Match plot heights to dashboard height after both plots are created
    setTimeout(() => {
        const dashboard = document.querySelector('#covertDatacentersTopSection .dashboard');
        const plotContainers = document.querySelectorAll('#covertDatacentersTopSection .plot-container');
        if (dashboard && plotContainers.length > 0) {
            const dashboardHeight = dashboard.offsetHeight;
            plotContainers.forEach(container => {
                container.style.height = dashboardHeight + 'px';
            });
            // Force resize after setting height
            setTimeout(() => {
                Plotly.Plots.resize('datacenterCapacityCcdfPlot');
                Plotly.Plots.resize('datacenterCombinedPlot');
            }, 50);
        }
    }, 150);
}

function plotDatacenterCapacityCcdf(data) {
    if (!data.black_datacenters || !data.black_datacenters.capacity_ccdfs) {
        document.getElementById('datacenterCapacityCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    // Only plot 1x threshold with green color
    const ccdf = data.black_datacenters.capacity_ccdfs[1];
    if (!ccdf || ccdf.length === 0) {
        document.getElementById('datacenterCapacityCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    const traces = [{
        x: ccdf.map(d => d.x),
        y: ccdf.map(d => d.y),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#5AA89B', width: 2 },  // Green color (previously used for 5x)
        name: '"Detection" = >1x update',
        hovertemplate: 'Capacity: %{x:.2f} GW<br>P(≥x): %{y:.3f}<extra></extra>'
    }];

    const layout = {
        xaxis: {
            title: "Datacenter Capacity (GW) Built Before 'Detection'",
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            type: 'log',
            automargin: true
        },
        yaxis: {
            title: 'P(capacity > x)',
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
        margin: { l: 55, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('datacenterCapacityCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('datacenterCapacityCcdfPlot'), 50);
}

function plotDatacenterLR(data) {
    if (!data.black_datacenters || !data.black_datacenters.lr_datacenters) return;

    const cd = data.black_datacenters;
    const years = cd.years;

    // Color for LR
    const lrColor = '#5B8DBE'; // Blue for likelihood ratio

    // Create traces with shaded percentile band
    const traces = [
        // 25th percentile (invisible, used for fill)
        {
            x: years,
            y: cd.lr_datacenters.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
        },
        // 75th percentile (fills to previous trace)
        {
            x: years,
            y: cd.lr_datacenters.p75,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: lrColor + '30',
            line: { color: 'transparent' },
            name: '25th-75th %ile',
            hovertemplate: 'LR: %{y:.2f}<extra></extra>'
        },
        // Median line
        {
            x: years,
            y: cd.lr_datacenters.median,
            type: 'scatter',
            mode: 'lines',
            line: { color: lrColor, width: 3 },
            name: 'Median LR',
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
            title: 'Likelihood Ratio',
            titlefont: { size: 13, color: 'black' },
            tickfont: { size: 10, color: 'black' },
            automargin: true,
            type: 'log'
        },
        showlegend: true,
        hovermode: 'x unified',
        margin: { l: 50, r: 20, t: 10, b: 85, pad: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: {
            orientation: 'h',
            y: -0.2,
            x: 0.5,
            xanchor: 'center',
            font: { size: 10 }
        }
    };

    Plotly.newPlot('datacenterLRPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('datacenterLRPlot'), 50);
}

function updateDatacenterDashboard(data) {
    // Populate datacenter dashboard
    if (data.black_datacenters && data.black_datacenters.individual_capacity_before_detection && data.black_datacenters.individual_time_before_detection) {
        const capacities = data.black_datacenters.individual_capacity_before_detection;
        const times = data.black_datacenters.individual_time_before_detection;

        // Calculate 50th percentile (median)
        const sortedCapacities = [...capacities].sort((a, b) => a - b);
        const sortedTimes = [...times].sort((a, b) => a - b);
        const p80Capacity = sortedCapacities[Math.floor(sortedCapacities.length * 0.5)];
        const p80Time = sortedTimes[Math.floor(sortedTimes.length * 0.5)];

        // Display capacity with proper formatting
        if (p80Capacity >= 1) {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${p80Capacity.toFixed(1)} GW`;
        } else if (p80Capacity >= 0.001) {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${(p80Capacity * 1000).toFixed(0)} MW`;
        } else {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${(p80Capacity * 1000).toFixed(1)} MW`;
        }

        // Display time
        document.getElementById('dashboardDatacenterTime').textContent = p80Time.toFixed(1);
    }
}

function populateUnconcealedDatacenterBreakdown(data) {
    // Populate the unconcealed datacenter capacity breakdown boxes using backend values

    // Get all values from backend data
    const bd = data.black_datacenters || {};
    const years = bd.prc_capacity_years || [];
    const capacityGw = bd.prc_capacity_gw || [];
    const capacityAtAgreementYear = bd.prc_capacity_at_agreement_year_gw ?? 0;
    const fractionDiverted = bd.fraction_diverted ?? 0.01;
    const covertUnconcealedCapacity = capacityAtAgreementYear * fractionDiverted;

    // Get agreement year from backend data (last year in the trajectory)
    const agreementYear = years.length > 0 ? years[years.length - 1] : 2030;

    // Update all agreement year display spans
    document.querySelectorAll('.agreement-year-display').forEach(el => {
        el.textContent = agreementYear;
    });

    // Format helper
    const formatGW = (gw) => {
        if (gw >= 1) {
            return `${gw.toFixed(1)} GW`;
        } else {
            return `${(gw * 1000).toFixed(0)} MW`;
        }
    };

    // Plot PRC datacenter capacity over time using backend data
    plotPrcDatacenterCapacityOverTime(data, agreementYear);

    // Populate the boxes with backend values
    document.getElementById('totalPrcDatacenterCapacityNotConcealedDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${formatGW(capacityAtAgreementYear)}</div>
            <div class="breakdown-label">Total PRC datacenter capacity<br>not built for concealment in <span class="agreement-year-display">${agreementYear}</span></div>
        </div>`;

    document.getElementById('proportionUnconcealedDivertedDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${(fractionDiverted * 100).toFixed(0)}%</div>
            <div class="breakdown-label">Proportion of unconcealed<br>capacity diverted to covert project</div>
        </div>`;

    document.getElementById('covertDatacenterCapacityUnconcealedDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${formatGW(covertUnconcealedCapacity)}</div>
            <div class="breakdown-label">Covert datacenter capacity<br>not built for concealment</div>
        </div>`;

    // Add hover effects and click handlers for the clickable boxes
    const clickableBoxes = [
        { id: 'totalPrcDatacenterCapacityNotConcealedDisplay', inputId: 'total_prc_compute_stock_in_2025' },
        { id: 'proportionUnconcealedDivertedDisplay', inputId: 'fraction_of_datacenter_capacity_not_built_for_concealment_diverted' }
    ];

    clickableBoxes.forEach(({ id, inputId }) => {
        const boxElement = document.getElementById(id);
        if (boxElement) {
            const innerElement = boxElement.querySelector('.breakdown-box-inner');
            if (innerElement) {
                innerElement.style.transition = 'all 0.2s ease';
                innerElement.addEventListener('mouseenter', () => {
                    innerElement.style.boxShadow = '0 0 6px rgba(0, 123, 255, 0.25)';
                    innerElement.style.transform = 'scale(1.015)';
                });
                innerElement.addEventListener('mouseleave', () => {
                    innerElement.style.boxShadow = '';
                    innerElement.style.transform = '';
                });
            }
            boxElement.style.cursor = 'pointer';
            boxElement.addEventListener('click', () => {
                const inputElement = document.getElementById(inputId);
                if (inputElement) {
                    inputElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    setTimeout(() => {
                        inputElement.focus();
                        inputElement.select();
                    }, 300);
                }
            });
        }
    });
}

function plotPrcDatacenterCapacityOverTime(data, agreementYear) {
    // Get data from backend
    const bd = data.black_datacenters || {};
    const years = bd.prc_capacity_years || [];
    const capacityGw = bd.prc_capacity_gw || [];

    if (years.length === 0 || capacityGw.length === 0) {
        console.warn('No PRC datacenter capacity data available');
        return;
    }

    const totalCapacityAtAgreementYear = capacityGw[capacityGw.length - 1];

    const traces = [
        // Shaded region for capacity not built for concealment
        {
            x: years,
            y: capacityGw,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(91, 141, 190, 0.2)',
            line: { color: 'transparent' },
            name: 'Capacity of datacenters not built for concealment',
            hovertemplate: '%{y:.2f} GW<extra>Not built for concealment</extra>'
        },
        // Total PRC compute stock energy consumption line
        {
            x: years,
            y: capacityGw,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#2B5B8A', width: 2 },
            name: 'Total PRC compute energy consumption',
            hovertemplate: '%{y:.2f} GW<extra>Total PRC energy</extra>'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true
        },
        yaxis: {
            title: 'GW',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
            automargin: true
        },
        margin: { l: 50, r: 20, t: 15, b: 60 },
        height: 250,
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            font: { size: 9 },
            bgcolor: 'rgba(255,255,255,0.8)'
        },
        annotations: [],
        shapes: [{
            type: 'line',
            x0: agreementYear,
            y0: 0,
            x1: agreementYear,
            y1: totalCapacityAtAgreementYear,
            line: {
                color: '#999',
                width: 1,
                dash: 'dot'
            }
        }]
    };

    Plotly.newPlot('prcDatacenterCapacityOverTimePlot', traces, layout, {responsive: true, displayModeBar: false});
    setTimeout(() => Plotly.Plots.resize('prcDatacenterCapacityOverTimePlot'), 50);
}

function populateDatacenterCapacityBreakdown(data) {
    // Populate the breakdown boxes with parameter values from input elements

    // Format helper
    const formatGW = (gw) => {
        if (gw >= 1) {
            return `${gw.toFixed(1)} GW`;
        } else {
            return `${(gw * 1000).toFixed(0)} MW`;
        }
    };

    // Get covert unconcealed capacity from backend data and populate the box
    const bd = data.black_datacenters || {};
    const capacityAtAgreementYear = bd.prc_capacity_at_agreement_year_gw ?? 0;
    const fractionDiverted = bd.fraction_diverted ?? 0.01;
    const covertUnconcealedCapacity = capacityAtAgreementYear * fractionDiverted;

    document.getElementById('covertUnconcealedCapacityDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${formatGW(covertUnconcealedCapacity)}</div>
            <div class="breakdown-label">Covert datacenter capacity<br>not built for concealment</div>
        </div>`;

    // Get total PRC energy consumption from input
    const totalPrcEnergyInput = document.getElementById('total_GW_of_PRC_energy_consumption');
    const totalPrcEnergy = totalPrcEnergyInput ? parseFloat(totalPrcEnergyInput.value) : 1000;
    const totalPrcEnergyFormatted = totalPrcEnergy >= 1000 ? `${(totalPrcEnergy / 1000).toFixed(0)}K GW` : `${totalPrcEnergy} GW`;
    document.getElementById('totalPrcEnergyDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${totalPrcEnergyFormatted}</div>
            <div class="breakdown-label">PRC energy<br>consumption</div>
        </div>`;

    // Get max proportion for covert datacenters from input
    const maxProportionInput = document.getElementById('max_proportion_of_PRC_energy_consumption');
    const maxProportion = maxProportionInput ? parseFloat(maxProportionInput.value) : 0.01;
    const maxProportionPercent = Math.round(maxProportion * 100);
    document.getElementById('maxProportionEnergyDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${maxProportionPercent}%</div>
            <div class="breakdown-label">Max % energy</div>
        </div>`;

    // Update inline text in description (there may be multiple elements)
    const inlineElements = document.querySelectorAll('.inlineMaxEnergyPercent');
    inlineElements.forEach(el => {
        el.textContent = `${maxProportionPercent}%`;
    });

    // Update the maxEnergyProportionText span in the section description
    const maxEnergyProportionText = document.getElementById('maxEnergyProportionText');
    if (maxEnergyProportionText) {
        maxEnergyProportionText.textContent = maxProportionPercent;
    }

    // Get datacenter construction workers from input
    const workersInput = document.getElementById('datacenter_construction_labor');
    const workers = workersInput ? parseInt(workersInput.value) : 10000;
    const workersFormatted = workers >= 1000 ? `${(workers / 1000).toFixed(0)}K` : workers.toLocaleString();
    document.getElementById('datacenterWorkersDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${workersFormatted}</div>
            <div class="breakdown-label">Construction<br>workers</div>
        </div>`;

    // Get MW per worker per year from input
    const mwPerWorkerInput = document.getElementById('MW_per_construction_worker_per_year');
    const mwPerWorker = mwPerWorkerInput ? parseFloat(mwPerWorkerInput.value) : 0.2;
    document.getElementById('mwPerWorkerDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${mwPerWorker.toFixed(2)}</div>
            <div class="breakdown-label">MW built per<br>worker per year</div>
        </div>`;

    // Get years before agreement year from input, then calculate the actual start year
    const yearsBeforeInput = document.getElementById('years_before_agreement_year_prc_starts_building_black_datacenters');
    const agreementYearInput = document.getElementById('agreement_year');
    const agreementYear = agreementYearInput ? parseInt(agreementYearInput.value) : 2030;
    const yearsBefore = yearsBeforeInput ? parseInt(yearsBeforeInput.value) : 0;
    const datacenterStartYear = agreementYear - yearsBefore;
    document.getElementById('agreementYearDisplay').innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div class="breakdown-box-inner">${datacenterStartYear}</div>
            <div class="breakdown-label">Year PRC starts building<br>covert datacenters</div>
        </div>`;

    // Add hover effects to the boxes
    const boxes = [
        document.querySelector('#totalPrcEnergyDisplay .breakdown-box-inner'),
        document.querySelector('#maxProportionEnergyDisplay .breakdown-box-inner'),
        document.querySelector('#datacenterWorkersDisplay .breakdown-box-inner'),
        document.querySelector('#mwPerWorkerDisplay .breakdown-box-inner'),
        document.querySelector('#agreementYearDisplay .breakdown-box-inner')
    ];
    boxes.forEach(box => {
        if (box) {
            box.style.transition = 'all 0.2s ease';
            box.addEventListener('mouseenter', () => {
                box.style.boxShadow = '0 0 6px rgba(0, 123, 255, 0.25)';
                box.style.transform = 'scale(1.015)';
            });
            box.addEventListener('mouseleave', () => {
                box.style.boxShadow = '';
                box.style.transform = '';
            });
        }
    });

    // Plot datacenter capacity over time
    if (data.black_project_model && data.black_project_model.datacenter_capacity) {
        const years = data.black_project_model.years;
        const capacity_median = data.black_project_model.datacenter_capacity.median;
        const capacity_p25 = data.black_project_model.datacenter_capacity.p25;
        const capacity_p75 = data.black_project_model.datacenter_capacity.p75;

        const traces = [
            // Upper bound of shaded region (invisible line)
            {
                x: years,
                y: capacity_p75,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Lower bound with fill to previous trace
            {
                x: years,
                y: capacity_p25,
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(90, 168, 155, 0.2)',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Median line
            {
                x: years,
                y: capacity_median,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#2D6B61', width: 3 },  // Dark turquoise
                name: 'Median',
                showlegend: false
            }
        ];

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true
            },
            yaxis: {
                title: 'GW',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true
            },
            margin: { l: 50, r: 20, t: 15, b: 60 },
            height: 250,
            hovermode: 'x unified',
        };

        layout.xaxis.range = [years[0], years[years.length - 1]];

        Plotly.newPlot('datacenterCapacityBreakdownPlot', traces, layout, {responsive: true, displayModeBar: false});
        setTimeout(() => Plotly.Plots.resize('datacenterCapacityBreakdownPlot'), 50);
    }
}
