// P(Catastrophe) Dashboard Component JavaScript

// Format a probability as a percentage with color coding
function formatProbability(p, colorScale = 'danger') {
    const pct = (p * 100).toFixed(1);
    let color;
    if (colorScale === 'danger') {
        // Higher is worse (red)
        if (p < 0.1) color = '#27ae60';
        else if (p < 0.2) color = '#f39c12';
        else if (p < 0.3) color = '#e67e22';
        else color = '#c0392b';
    } else {
        color = '#333';
    }
    return `<span style="color: ${color}; font-weight: 700; font-size: 24px;">${pct}%</span>`;
}

// Render the stats for a trajectory
function renderTrajectoryStats(data, containerId) {
    const container = document.getElementById(containerId);

    if (!data) {
        container.innerHTML = '<p style="color: #888; font-style: italic;">No data available (milestones not reached)</p>';
        return;
    }

    const html = `
        <div style="display: grid; gap: 15px;">
            <!-- Milestone times -->
            <div style="display: flex; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid #ddd;">
                <div>
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">AC Reached</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333;">${data.ac_time.toFixed(2)}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">SAR Reached</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333;">${data.sar_time.toFixed(2)}</div>
                </div>
            </div>

            <!-- Handoff durations -->
            <div style="display: flex; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid #ddd;">
                <div>
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">Handoff Duration</div>
                    <div style="font-size: 16px; font-weight: 500; color: #555;">${(data.handoff_duration_years * 12).toFixed(1)} months</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">Adjusted Duration</div>
                    <div style="font-size: 16px; font-weight: 500; color: #555;">${(data.adjusted_handoff_duration_years * 12).toFixed(1)} months</div>
                </div>
            </div>

            <!-- Probabilities -->
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">P(AI Takeover)</div>
                    ${formatProbability(data.p_ai_takeover)}
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">P(Human Power Grabs)</div>
                    ${formatProbability(data.p_human_power_grabs)}
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Render P(catastrophe) dashboard from the combined data response
function renderPCatastropheDashboard(data) {
    if (!data || !data.p_catastrophe) {
        console.warn('No P(catastrophe) data available');
        return;
    }
    renderTrajectoryStats(data.p_catastrophe.global, 'globalTrajectoryStats');
    renderTrajectoryStats(data.p_catastrophe.covert, 'covertTrajectoryStats');
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { formatProbability, renderTrajectoryStats, renderPCatastropheDashboard };
}
