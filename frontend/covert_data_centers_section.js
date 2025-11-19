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
        // Capacity: Upper bound (invisible)
        {
            x: years,
            y: dcm.datacenter_capacity.p75.map(v => v * 1000),
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // Capacity: Lower bound (fills to previous)
        {
            x: years,
            y: dcm.datacenter_capacity.p25.map(v => v * 1000),
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(90, 168, 155, 0.2)',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y'
        },
        // Capacity: Median line
        {
            x: years,
            y: dcm.datacenter_capacity.median.map(v => v * 1000),
            type: 'scatter',
            mode: 'lines',
            line: { color: capacityColor, width: 3 },
            name: 'Covert Datacenter capacity',
            hovertemplate: 'Capacity: %{y:.0f} MW<extra></extra>',
            yaxis: 'y'
        },
        // LR: 25th percentile (invisible)
        {
            x: years,
            y: cd.lr_datacenters.p25,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
            yaxis: 'y2'
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
            yaxis: 'y2'
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
            yaxis: 'y2'
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 13 },
            automargin: true,
            showline: false,
            mirror: false
        },
        yaxis: {
            title: 'Energy capacity (MW)',
            titlefont: { size: 13, color: capacityColor },
            tickfont: { size: 10, color: capacityColor },
            automargin: true,
            side: 'left',
            showline: false,
            mirror: false
        },
        yaxis2: {
            title: 'Evidence (LR)',
            titlefont: { size: 13, color: lrColor },
            tickfont: { size: 10, color: lrColor },
            overlaying: 'y',
            side: 'right',
            type: 'log',
            automargin: true,
            showline: false,
            mirror: false
        },
        showlegend: false,
        hovermode: 'x unified',
        margin: { l: 60, r: 60, t: 10, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('datacenterCombinedPlot', traces, layout, {displayModeBar: false, responsive: true});
    setTimeout(() => Plotly.Plots.resize('datacenterCombinedPlot'), 50);
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
            automargin: true,
            showline: false,
            mirror: false
        },
        yaxis: {
            title: 'P(capacity > x)',
            titlefont: { size: 13 },
            tickfont: { size: 10 },
            range: [0, 1],
            automargin: true,
            showline: false,
            mirror: false
        },
        showlegend: false,
        hovermode: 'closest',
        margin: { l: 50, r: 10, t: 10, b: 65, pad: 10 },
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

        // Calculate median (50th percentile)
        const sortedCapacities = [...capacities].sort((a, b) => a - b);
        const sortedTimes = [...times].sort((a, b) => a - b);
        const medianCapacity = sortedCapacities[Math.floor(sortedCapacities.length / 2)];
        const medianTime = sortedTimes[Math.floor(sortedTimes.length / 2)];

        // Display capacity with proper formatting
        if (medianCapacity >= 1) {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${medianCapacity.toFixed(1)} GW`;
        } else if (medianCapacity >= 0.001) {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${(medianCapacity * 1000).toFixed(0)} MW`;
        } else {
            document.getElementById('dashboardDatacenterCapacity').textContent = `${(medianCapacity * 1000).toFixed(1)} MW`;
        }

        // Display time
        document.getElementById('dashboardDatacenterTime').textContent = `${medianTime.toFixed(1)} years`;
    }
}
