// JavaScript for Covert Data Centers section

function plotDatacenterCombined(data) {
    if (!data.dark_compute_model || !data.dark_compute_model.datacenter_capacity) return;
    if (!data.covert_datacenters || !data.covert_datacenters.lr_datacenters) return;

    const dcm = data.dark_compute_model;
    const cd = data.covert_datacenters;
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
            title: {
                text: 'Evidence (LR)',
                standoff: 15
            },
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: lrColor },
            automargin: true,
            side: 'left',
            type: 'log'
        },
        yaxis2: {
            title: {
                text: 'Capacity (GW)',
                standoff: 15
            },
            titlefont: { size: 13, color: '#000' },
            tickfont: { size: 10, color: capacityColor },
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
    if (!data.covert_datacenters || !data.covert_datacenters.capacity_ccdfs) {
        document.getElementById('datacenterCapacityCcdfPlot').innerHTML = '<p>No detection data available</p>';
        return;
    }

    // Only plot 1x threshold with green color
    const ccdf = data.covert_datacenters.capacity_ccdfs[1];
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
        margin: { l: 55, r: 0, t: 0, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('datacenterCapacityCcdfPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('datacenterCapacityCcdfPlot'), 50);
}

function plotDatacenterLR(data) {
    if (!data.covert_datacenters || !data.covert_datacenters.lr_datacenters) return;

    const cd = data.covert_datacenters;
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
    if (data.covert_datacenters && data.covert_datacenters.individual_capacity_before_detection && data.covert_datacenters.individual_time_before_detection) {
        const capacities = data.covert_datacenters.individual_capacity_before_detection;
        const times = data.covert_datacenters.individual_time_before_detection;

        // Calculate 80th percentile
        const sortedCapacities = [...capacities].sort((a, b) => a - b);
        const sortedTimes = [...times].sort((a, b) => a - b);
        const p80Capacity = sortedCapacities[Math.floor(sortedCapacities.length * 0.8)];
        const p80Time = sortedTimes[Math.floor(sortedTimes.length * 0.8)];

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
