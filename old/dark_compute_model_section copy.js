// JavaScript for Dark Compute Model section

function plotDarkComputeModel(data) {
    // Plot dark compute equation section

    // Plot 1: Initial dark compute (histogram)
    if (data.initial_stock && data.initial_stock.initial_compute_stock_samples) {
        // Use raw numbers with k/M formatting
        const samples = data.initial_stock.initial_compute_stock_samples;
        plotPDF('initialDarkComputePlot', samples, '#9B72B0', 'H100 Equivalents');
    }

    // Plot 2: Flow from covert fab (line plot - cumulative production over time)
    if (data.dark_compute_model && data.dark_compute_model.years) {
        const years = data.dark_compute_model.years;
        // Use raw numbers with k/M formatting - includes ALL simulations (not just where fab is built)
        const h100e_median = data.dark_compute_model.covert_fab_flow_all_sims.median;
        const h100e_p25 = data.dark_compute_model.covert_fab_flow_all_sims.p25;
        const h100e_p75 = data.dark_compute_model.covert_fab_flow_all_sims.p75;

        // Create traces for the shaded region and median line
        const traces = [
            // Upper bound of shaded region (invisible line)
            {
                x: years,
                y: h100e_p75,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Lower bound with fill to previous trace
            {
                x: years,
                y: h100e_p25,
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(187, 143, 206, 0.2)',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Median line
            {
                x: years,
                y: h100e_median,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#8E44AD', width: 2 },
                name: 'Covert Fab Production',
                showlegend: true
            }
        ];

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                range: [years[0], 2037]
            },
            yaxis: {
                title: 'H100e',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                tickformat: '.2s'
            },
            margin: { l: 50, r: 20, t: 35, b: 40 },
            height: 250,
            hovermode: 'x unified',
            legend: {
                x: 0.5,
                y: 1.0,
                xanchor: 'center',
                yanchor: 'bottom',
                orientation: 'h',
                font: { size: 10 }
            },
            showlegend: true
        };

        Plotly.newPlot('covertFabFlowPlot', traces, layout, {responsive: true});
    }

    // Plot 3: Average compute survival rate
    if (data.dark_compute_model && data.dark_compute_model.survival_rate) {
        const years = data.dark_compute_model.years;
        const survival_rate_median = data.dark_compute_model.survival_rate.median;
        const survival_rate_p25 = data.dark_compute_model.survival_rate.p25;
        const survival_rate_p75 = data.dark_compute_model.survival_rate.p75;

        const traces = [
            // Upper bound of shaded region (invisible line)
            {
                x: years,
                y: survival_rate_p75,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Lower bound with fill to previous trace
            {
                x: years,
                y: survival_rate_p25,
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(231, 76, 60, 0.2)',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Median line
            {
                x: years,
                y: survival_rate_median,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#E74C3C', width: 2 },
                name: 'Median',
                showlegend: false
            }
        ];

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                range: [years[0], 2037]
            },
            yaxis: {
                title: 'Surviving Fraction',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                range: [0, 1]
            },
            margin: { l: 50, r: 20, t: 10, b: 40 },
            height: 250,
            hovermode: 'x unified'
        };

        Plotly.newPlot('chipSurvivalPlot', traces, layout, {responsive: true});
    }

    // Plot 4: Dark compute (surviving, not limited by capacity)
    if (data.dark_compute_model && data.dark_compute_model.total_dark_compute) {
        const years = data.dark_compute_model.years;
        const dark_compute_median = data.dark_compute_model.total_dark_compute.median;
        const dark_compute_p25 = data.dark_compute_model.total_dark_compute.p25;
        const dark_compute_p75 = data.dark_compute_model.total_dark_compute.p75;

        const traces = [
            // Upper bound of shaded region (invisible line)
            {
                x: years,
                y: dark_compute_p75,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Lower bound with fill to previous trace
            {
                x: years,
                y: dark_compute_p25,
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(142, 68, 173, 0.2)',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Median line
            {
                x: years,
                y: dark_compute_median,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#8E44AD', width: 2 },
                name: 'Median',
                showlegend: false
            }
        ];

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 }
            },
            yaxis: {
                title: 'H100e',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                tickformat: '.2s'
            },
            margin: { l: 50, r: 20, t: 10, b: 40 },
            height: 250,
            hovermode: 'x unified'
        };

        layout.xaxis.range = [years[0], 2037];

        Plotly.newPlot('totalDarkComputePlot', traces, layout, {responsive: true});
    }

    // Calculate shared Y-axis range for both datacenter capacity and energy plots
    let sharedYMax = 0;
    if (data.dark_compute_model && data.dark_compute_model.datacenter_capacity && data.dark_compute_model.dark_compute_energy) {
        const capacity_p75 = data.dark_compute_model.datacenter_capacity.p75;
        const capacity_median = data.dark_compute_model.datacenter_capacity.median;
        const capacity_p25 = data.dark_compute_model.datacenter_capacity.p25;
        const energyData = data.dark_compute_model.dark_compute_energy;
        const datacenterCapacity = data.dark_compute_model.datacenter_capacity.median;

        // Calculate total energy at each time point
        const totalEnergy = energyData.map(yearData => yearData[0] + yearData[1]);

        // Find max across both datasets
        const capacityYValues = [...capacity_p75, ...capacity_median, ...capacity_p25];
        const energyYValues = [...totalEnergy, ...datacenterCapacity];
        const allYValues = [...capacityYValues, ...energyYValues];
        const maxY = Math.max(...allYValues);
        sharedYMax = maxY * 1.8;
    }

    // Plot 5: Datacenter capacity (GW)
    if (data.dark_compute_model && data.dark_compute_model.datacenter_capacity) {
        const years = data.dark_compute_model.years;
        const capacity_median = data.dark_compute_model.datacenter_capacity.median;
        const capacity_p25 = data.dark_compute_model.datacenter_capacity.p25;
        const capacity_p75 = data.dark_compute_model.datacenter_capacity.p75;

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

        // Use shared Y-axis range
        const yMax = sharedYMax;

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
                automargin: true,
                range: [0, yMax]
            },
            margin: { l: 50, r: 50, t: 10, b: 55, pad: 10 },
            height: 250,
            hovermode: 'x unified'
        };

        layout.xaxis.range = [years[0], 2037];

        Plotly.newPlot('datacenterCapacityPlot2', traces, layout, {responsive: true, displayModeBar: false});
        setTimeout(() => Plotly.Plots.resize('datacenterCapacityPlot2'), 50);
    }

    // Energy Consumption Stacked Area Plot (for median simulation)
    // Shows two sources: Initial Stock (bottom) and Fab-Produced (top)
    if (data.dark_compute_model && data.dark_compute_model.dark_compute_energy) {
        const years = data.dark_compute_model.years;
        const energyData = data.dark_compute_model.dark_compute_energy;
        const sourceLabels = data.dark_compute_model.energy_source_labels;
        const datacenterCapacity = data.dark_compute_model.datacenter_capacity.median;

        // Two colors: turquoise shades for initial stock (bottom) and fab-produced (top)
        const colors = ['#4A9B8E', '#74B3A8'];  // Dark turquoise, light turquoise

        const traces = [];

        // Check if fab has any energy (index 1 is fab)
        const fabEnergy = energyData.map(yearData => yearData[1]);
        const hasFabEnergy = fabEnergy.some(val => val > 0);

        // Calculate total energy (sum of both sources) at each time point
        const totalEnergy = energyData.map(yearData => yearData[0] + yearData[1]);

        // LAYER 1: Create stacked area traces for each source (bottom layer)
        for (let i = 0; i < sourceLabels.length; i++) {
            const energyAtSource = energyData.map(yearData => yearData[i]);
            const sourceLabel = sourceLabels[i];

            // Skip fab compute (i=1) if there's no fab energy in the median
            if (i === 1 && !hasFabEnergy) {
                continue;
            }

            traces.push({
                x: years,
                y: energyAtSource,
                type: 'scatter',
                mode: 'lines',
                stackgroup: 'energy',
                fillcolor: colors[i],
                line: { width: 0 },
                name: sourceLabel,
                hovertemplate: `${sourceLabel}<br>Energy: %{y:.2f} GW<extra></extra>`
            });
        }

        // LAYER 2: Create hash pattern ONLY where total energy exceeds capacity
        // For each time point, the unpowered region is: max(0, totalEnergy - capacity)
        // We'll stack this on TOP of the capacity line
        const unpoweredEnergy = years.map((year, i) =>
            Math.max(0, totalEnergy[i] - datacenterCapacity[i])
        );

        // Create the hash region as a shape that sits above the capacity line
        // It goes from capacity to (capacity + unpowered)
        const hashX = [...years, ...years.slice().reverse()];
        const hashYTop = years.map((year, i) => datacenterCapacity[i] + unpoweredEnergy[i]);
        const hashY = [...hashYTop, ...datacenterCapacity.slice().reverse()];

        traces.push({
            x: hashX,
            y: hashY,
            type: 'scatter',
            mode: 'none',
            fill: 'toself',
            fillcolor: 'rgba(0, 0, 0, 0)',
            fillpattern: {
                shape: '/',
                fgcolor: 'rgba(100, 100, 100, 0.8)',
                bgcolor: 'rgba(0, 0, 0, 0)',
                size: 8,
                solidity: 0.7
            },
            showlegend: false,
            hoverinfo: 'skip',
            line: { width: 0 }
        });

        // LAYER 3: Add datacenter capacity line on top (just a line, no hash)
        traces.push({
            x: years,
            y: datacenterCapacity,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#2D6B61', width: 3 },
            name: 'Covert Datacenter Capacity',
            hovertemplate: 'Capacity: %{y:.1f} GW<extra></extra>'
        });

        // Add legend entry for the hash pattern
        traces.push({
            x: [years[0], years[1]],
            y: [null, null],
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 0, 0, 0)',
            fillpattern: {
                shape: '/',
                fgcolor: 'rgba(100, 100, 100, 0.8)',
                bgcolor: 'rgba(0, 0, 0, 0)',
                size: 8,
                solidity: 0.7
            },
            line: { width: 0 },
            name: 'Cannot be operated',
            showlegend: true,
            hoverinfo: 'skip'
        });

        // Datacenter capacity line already added above

        // Use shared Y-axis range
        const yMax = sharedYMax;

        const layout = {
            xaxis: {
                title: 'Year',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true
            },
            yaxis: {
                title: 'Energy Usage',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true,
                range: [0, yMax]
            },
            margin: { l: 50, r: 50, t: 10, b: 55, pad: 10 },
            height: 250,
            hovermode: 'x unified',
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ccc',
                borderwidth: 1,
                font: { size: 7 }
            },
            autosize: true
        };

        layout.xaxis.range = [years[0], 2037];

        Plotly.newPlot('darkComputeEnergyPlot', traces, layout, {responsive: true, displayModeBar: false});
        setTimeout(() => Plotly.Plots.resize('darkComputeEnergyPlot'), 50);
    }

    // Plot 6: Operational dark compute (limited by capacity)
    if (data.dark_compute_model && data.dark_compute_model.operational_dark_compute) {
        const years = data.dark_compute_model.years;
        const operational_median = data.dark_compute_model.operational_dark_compute.median;
        const operational_p25 = data.dark_compute_model.operational_dark_compute.p25;
        const operational_p75 = data.dark_compute_model.operational_dark_compute.p75;

        const traces = [
            // Upper bound of shaded region (invisible line)
            {
                x: years,
                y: operational_p75,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'transparent' },
                showlegend: false,
                hoverinfo: 'skip'
            },
            // Lower bound with fill to previous trace
            {
                x: years,
                y: operational_p25,
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
                y: operational_median,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#5AA89B', width: 2 },
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
                title: 'H100e',
                titlefont: { size: 10 },
                tickfont: { size: 9 },
                automargin: true,
                tickformat: '.2s'
            },
            margin: { l: 50, r: 50, t: 10, b: 55, pad: 10 },
            height: 250,
            hovermode: 'x unified'
        };

        layout.xaxis.range = [years[0], 2037];

        Plotly.newPlot('operationalDarkComputePlot', traces, layout, {responsive: true, displayModeBar: false});
        setTimeout(() => Plotly.Plots.resize('operationalDarkComputePlot'), 50);
    }
}
