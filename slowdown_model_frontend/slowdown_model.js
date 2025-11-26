// JavaScript for AI Takeoff Trajectory

// Fetch data from backend and plot
async function loadAndPlotTakeoffModel() {
    try {
        const response = await fetch('/get_slowdown_model_data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        plotTakeoffModel(data);
    } catch (error) {
        console.error('Error loading takeoff model data:', error);
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">Error loading data. Please refresh the page.</p>';
    }
}

function plotTakeoffModel(data) {
    if (!data.takeoff_trajectories) {
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">No trajectory data available</p>';
        return;
    }

    const trajectories = data.takeoff_trajectories;
    const milestones_global = trajectories.milestones_global;
    const trajectory_times = trajectories.trajectory_times;
    const global_ai_speedup = trajectories.global_ai_speedup;

    if (!milestones_global || !trajectory_times || !global_ai_speedup) {
        document.getElementById('slowdownPlot').innerHTML = '<p style="text-align: center; color: #e74c3c;">Trajectory prediction failed - missing AI speedup data</p>';
        return;
    }

    // Get agreement start year from the data
    const agreement_year = data.agreement_year;

    // Define which milestones to display
    const milestoneNames = {
        'AC': 'AC',
        'SAR-level-experiment-selection-skill': 'SAR',
        'SIAR-level-experiment-selection-skill': 'SIAR',
        'TED-AI': 'TED-AI',
        'ASI': 'ASI'
    };

    // Extract milestone data with actual speedup values from progress_multiplier
    const milestoneData = [];
    for (const [key, label] of Object.entries(milestoneNames)) {
        if (milestones_global[key]) {
            const time = milestones_global[key].time;
            const progress_multiplier = milestones_global[key].progress_multiplier;

            if (progress_multiplier && !isNaN(progress_multiplier)) {
                milestoneData.push({
                    key: key,
                    label: label,
                    time: time,
                    speedup: progress_multiplier
                });
            }
        }
    }

    // Filter trajectory data to start from agreement year
    // and end a bit after ASI
    const asiTime = milestones_global['ASI']?.time || trajectory_times[trajectory_times.length - 1];
    const endYear = Math.min(trajectory_times[trajectory_times.length - 1], asiTime + 1);

    const filteredData = trajectory_times.map((time, i) => ({
        time,
        speedup: global_ai_speedup[i]
    })).filter(d => d.time >= agreement_year && d.time <= endYear);

    const traces = [
        {
            x: filteredData.map(d => d.time),
            y: filteredData.map(d => d.speedup),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#2A623D',
                width: 3
            },
            name: 'AI Software R&D Uplift',
            hovertemplate: 'Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>'
        },
        // Add milestone markers
        {
            x: milestoneData.map(m => m.time),
            y: milestoneData.map(m => m.speedup),
            type: 'scatter',
            mode: 'markers+text',
            marker: {
                color: '#2A623D',
                size: 10,
                line: {
                    color: 'white',
                    width: 2
                }
            },
            text: milestoneData.map(m => m.label),
            textposition: 'top center',
            textfont: {
                size: 11,
                color: '#2A623D',
                family: 'Arial, sans-serif'
            },
            hovertemplate: '%{text}<br>Year: %{x:.1f}<br>Speedup: %{y:.2f}x<extra></extra>',
            showlegend: false
        }
    ];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            automargin: true,
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        yaxis: {
            title: 'AI Software R&D Uplift (relative to human baseline)',
            titlefont: { size: 14 },
            tickfont: { size: 12 },
            automargin: true,
            type: 'log',
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        showlegend: false,
        hovermode: 'closest',
        margin: { l: 100, r: 40, t: 40, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#fafafa'
    };

    Plotly.newPlot('slowdownPlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Load data when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadAndPlotTakeoffModel();
});
