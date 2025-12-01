// Risk Breakdown Plot Component JavaScript

// Collect logs and send to backend
const _frontendLogs = [];
function _log(level, message, data) {
    _frontendLogs.push({ level, message, data, timestamp: new Date().toISOString() });
    // Also log to console
    if (data) {
        console[level](message, data);
    } else {
        console[level](message);
    }
}

function _flushLogs() {
    if (_frontendLogs.length === 0) return;
    const logsToSend = [..._frontendLogs];
    _frontendLogs.length = 0;
    fetch('/log_frontend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ logs: logsToSend })
    }).catch(err => console.error('Failed to send logs:', err));
}

// Flush logs periodically and on page unload
setInterval(_flushLogs, 2000);
window.addEventListener('beforeunload', _flushLogs);

/**
 * Format a number as a percentage string
 */
function formatPercent(val) {
    if (val === null || val === undefined) return '--%';
    return Math.round(val * 100) + '%';
}

/**
 * Format a number as years with 1 decimal place
 */
function formatYears(val) {
    if (val === null || val === undefined) return '-- yrs';
    return val.toFixed(1) + ' yrs';
}

/**
 * Format a number as a multiplier (e.g., "2.3x")
 */
function formatMultiplier(val) {
    if (val === null || val === undefined) return '--x';
    return val.toFixed(1) + 'x';
}

/**
 * Format a number with 2 decimal places
 */
function formatDecimal(val) {
    if (val === null || val === undefined) return '--';
    return val.toFixed(2);
}

/**
 * Safely set text content of an element by ID
 */
function setElementText(id, text) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = text;
    }
}

/**
 * Create a small mapping plot using Plotly
 * Styling copied from dark_compute_detection_section.js
 */
function createMappingPlot(containerId, xData, yData, xLabel, yLabel, currentX, currentY, color) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`createMappingPlot: Container not found: ${containerId}`);
        return;
    }

    // Validate data
    if (!xData || !yData || xData.length === 0 || yData.length === 0) {
        console.warn(`createMappingPlot: No data for ${containerId}`, { xData, yData });
        return;
    }

    const plotColor = color || '#5B8DBE';

    // Main curve
    const traces = [{
        x: xData,
        y: yData.map(v => v * 100),
        type: 'scatter',
        mode: 'lines',
        line: { color: plotColor, width: 2 },
        name: 'Curve',
        showlegend: false
    }];

    // Current point marker
    if (currentX != null && currentY != null && !isNaN(currentX) && !isNaN(currentY)) {
        traces.push({
            x: [currentX],
            y: [currentY * 100],
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: plotColor,
                size: 8,
                line: { color: 'white', width: 1 }
            },
            name: 'Current',
            showlegend: false,
            hovertemplate: `${xLabel}: %{x:.2f}<br>${yLabel}: %{y:.0f}%<extra></extra>`
        });
    }

    const layout = {
        xaxis: {
            title: { text: xLabel, font: { size: 11 } },
            type: 'log',
            tickfont: { size: 9 },
            gridcolor: 'rgba(128, 128, 128, 0.2)'
        },
        yaxis: {
            title: { text: yLabel, font: { size: 11 } },
            tickfont: { size: 9 },
            range: [0, 50],
            gridcolor: 'rgba(128, 128, 128, 0.2)'
        },
        showlegend: false,
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { l: 50, r: 20, t: 10, b: 50 },
        height: 200
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });
    setTimeout(() => Plotly.Plots.resize(containerId), 50);
}

/**
 * Update all Risk Breakdown values from backend data
 */
function updateRiskBreakdownValues(data) {
    console.log('updateRiskBreakdownValues called with data:', Object.keys(data));
    const rbd = data.risk_breakdown_data;
    if (!rbd) {
        console.warn('No risk_breakdown_data in response');
        return;
    }
    console.log('risk_breakdown_data keys:', Object.keys(rbd));

    // === Top-level probabilities (Section 0) ===
    // All "P(No X)" values are computed in the backend and sent directly
    // to ensure the math is consistent: P(No Domestic) = P(No AI) × P(No Human)
    setElementText('p-no-ai-takeover-value', formatPercent(rbd.p_ai_takeover));
    setElementText('p-no-human-power-grabs-value', formatPercent(rbd.p_human_power_grabs));

    // P(No Domestic Takeover) - display the actual product from the backend
    setElementText('p-no-domestic-takeover-value', formatPercent(rbd.p_no_domestic_takeover));

    // P(Domestic Takeover)
    setElementText('p-domestic-takeover-value', formatPercent(rbd.p_domestic_takeover));

    // === Section 1: P(Human Power Grabs) ===
    setElementText('sar-to-asi-duration', formatYears(rbd.sar_to_asi_duration));
    setElementText('p-human-power-grabs-result', formatPercent(rbd.p_human_power_grabs));

    // Anchor points for human power grabs
    if (rbd.anchor_points && rbd.anchor_points.human_power_grabs) {
        const hpg = rbd.anchor_points.human_power_grabs;
        setElementText('power-grab-anchor-1mo', formatPercent(hpg.t1));
        setElementText('power-grab-anchor-1yr', formatPercent(hpg.t2));
        setElementText('power-grab-anchor-10yr', formatPercent(hpg.t3));
    }

    // === Section 2: P(AI Takeover) breakdown ===
    // P(No Misalignment at Handoff)
    setElementText('p-no-misalignment-at-handoff-value', formatPercent(rbd.p_misalignment_at_handoff));

    // P(No Misalignment after Handoff)
    setElementText('p-no-misalignment-after-handoff-value', formatPercent(rbd.p_misalignment_after_handoff));

    // P(No AI Takeover) result (same value as section 0)
    setElementText('p-no-ai-takeover-result', formatPercent(rbd.p_no_ai_takeover));

    // P(AI Takeover) result
    setElementText('p-ai-takeover-result', formatPercent(rbd.p_ai_takeover));

    // === Section 3: Pre-handoff window ===
    setElementText('pre-ac-calendar-time', formatYears(rbd.pre_ac_calendar_time));
    setElementText('pre-ac-alignment-speedup', formatMultiplier(rbd.pre_ac_avg_alignment_speedup));
    setElementText('pre-ac-relevance', formatDecimal(rbd.relevance_discount));
    setElementText('pre-ac-adjusted-time', formatYears(rbd.pre_ac_adjusted_time));

    // Safety exponent displays
    const safetyExp = rbd.safety_speedup_exponent || 0.5;
    setElementText('safety-exponent-display-1', safetyExp.toFixed(1));
    setElementText('safety-exponent-display-2', safetyExp.toFixed(1));

    // Handoff speedup threshold displays
    const handoffThreshold = rbd.handoff_speedup_threshold || 20;
    setElementText('handoff-threshold-1', Math.round(handoffThreshold));
    setElementText('handoff-threshold-2', Math.round(handoffThreshold));
    setElementText('handoff-threshold-3', Math.round(handoffThreshold));

    // === Section 3: Handoff window ===
    setElementText('handoff-window-calendar-time', formatYears(rbd.handoff_window_calendar_time));
    setElementText('handoff-window-alignment-speedup', formatMultiplier(rbd.handoff_window_avg_alignment_speedup));
    setElementText('handoff-window-relevance', '1.0'); // Always 1.0 for handoff window
    setElementText('handoff-window-adjusted-time', formatYears(rbd.handoff_window_adjusted_time));

    // === Section 4: Slowdown effort adjustment ===
    setElementText('agreement-year-display', rbd.agreement_year ? Math.round(rbd.agreement_year) : '--');
    setElementText('agreement-year-display-2', rbd.agreement_year ? Math.round(rbd.agreement_year) : '--');
    setElementText('adjusted-time-before-agreement', formatYears(rbd.adjusted_time_before_agreement));
    setElementText('adjusted-time-during-slowdown', formatYears(rbd.adjusted_time_during_slowdown));
    setElementText('slowdown-effort-multiplier', formatMultiplier(rbd.slowdown_effort_multiplier));
    setElementText('total-adjusted-time-after-slowdown', formatYears(rbd.total_adjusted_alignment_time));

    // === Section 5: Total adjusted alignment time (same as section 4 result) ===
    setElementText('total-adjusted-alignment-time', formatYears(rbd.total_adjusted_alignment_time));

    // Anchor points for misalignment at handoff
    if (rbd.anchor_points && rbd.anchor_points.misalignment_at_handoff) {
        const mah = rbd.anchor_points.misalignment_at_handoff;
        setElementText('anchor-1mo', formatPercent(mah.t1));
        setElementText('anchor-1yr', formatPercent(mah.t2));
        setElementText('anchor-10yr', formatPercent(mah.t3));
    }

    // P(Misalignment at Handoff) result
    setElementText('p-misalignment-at-handoff-result', formatPercent(rbd.p_misalignment_at_handoff));

    // === Section 6: Post-handoff ===
    // Tax multiplier in description text
    const taxMult = rbd.tax_multiplier || 2.0;
    setElementText('tax-multiplier-inline', taxMult.toFixed(0));
    setElementText('tax-multiplier-direction', taxMult >= 1 ? 'higher' : 'lower');
    setElementText('tax-multiplier-display', taxMult.toFixed(0));
    setElementText('tax-multiplier-value', formatMultiplier(taxMult));

    // Post-handoff window calculation
    setElementText('post-handoff-calendar-time', formatYears(rbd.post_handoff_calendar_time));
    setElementText('post-handoff-calendar-time-2', formatYears(rbd.post_handoff_calendar_time));
    setElementText('post-handoff-avg-alignment-speedup', formatMultiplier(rbd.post_handoff_avg_alignment_speedup));
    setElementText('post-handoff-avg-capability-speedup', formatMultiplier(rbd.post_handoff_avg_capability_speedup));
    setElementText('safety-exponent-display-3', safetyExp.toFixed(1));

    // Post-handoff alignment tax calculation
    setElementText('post-handoff-alignment-time', formatYears(rbd.post_handoff_alignment_time));
    setElementText('post-handoff-alignment-time-2', formatYears(rbd.post_handoff_alignment_time));
    setElementText('post-handoff-capability-time', formatYears(rbd.post_handoff_capability_time));
    setElementText('post-handoff-capability-time-2', formatYears(rbd.post_handoff_capability_time));
    setElementText('alignment-tax-after-handoff', formatDecimal(rbd.alignment_tax_after_handoff));
    setElementText('alignment-tax-after-handoff-2', formatDecimal(rbd.alignment_tax_after_handoff));
    setElementText('effective-tax', formatDecimal(rbd.effective_tax));

    // P(Misalignment after Handoff) result
    setElementText('p-misalignment-after-handoff-result', formatPercent(rbd.p_misalignment_after_handoff));

    // === Create mapping plots ===
    console.log('Risk breakdown curves data:', rbd.curves);
    if (rbd.curves) {
        const durations = rbd.curves.durations;
        console.log('Durations:', durations ? durations.length : 'null');

        // Human power grabs mapping plot
        if (rbd.curves.human_power_grabs) {
            createMappingPlot(
                'human-power-grabs-mapping-plot',
                durations,
                rbd.curves.human_power_grabs,
                'SAR→ASI (years)',
                'P(Power Grab)',
                rbd.sar_to_asi_duration,
                rbd.p_human_power_grabs,
                '#5AA89B'
            );
        }

        // Misalignment at handoff mapping plot
        if (rbd.curves.misalignment_at_handoff) {
            createMappingPlot(
                'misalignment-mapping-plot',
                durations,
                rbd.curves.misalignment_at_handoff,
                'Adjusted Research (years)',
                'P(Misalignment)',
                rbd.total_adjusted_alignment_time,
                rbd.p_misalignment_at_handoff,
                '#E8A863'
            );
        }

        // Post-handoff mapping plot (using effective alignment tax as x-axis)
        if (rbd.curves.post_handoff && rbd.curves.post_handoff_x) {
            createMappingPlot(
                'post-handoff-mapping-plot',
                rbd.curves.post_handoff_x,  // Use separate x-axis for tax values
                rbd.curves.post_handoff,
                'Alignment tax paid',
                'P(Misalignment)',
                rbd.effective_tax,
                rbd.p_misalignment_after_handoff,
                '#E8A863'
            );
        }
    }
}

/**
 * Update the risk comparison dashboard with values at the end of the slowdown period
 * (Legacy function - kept for backward compatibility)
 */
function updateRiskComparisonDashboard(data) {
    const plotData = data.risk_reduction_over_time;

    if (!plotData || !plotData.slowdown_duration || plotData.slowdown_duration.length === 0) {
        return;
    }

    // Get the last values (at maximum slowdown duration)
    const lastIdx = plotData.slowdown_duration.length - 1;

    // AI Takeover risk
    const aiTakeoverNoSlowdown = plotData.p_ai_takeover_no_slowdown[lastIdx];
    const aiTakeoverSlowdown = plotData.p_ai_takeover_slowdown[lastIdx];

    // Human power grab risk
    const humanPowerGrabNoSlowdown = plotData.p_human_power_grabs_no_slowdown[lastIdx];
    const humanPowerGrabSlowdown = plotData.p_human_power_grabs_slowdown[lastIdx];

    // Update AI takeover dashboard element
    const aiTakeoverEl = document.getElementById('ai-takeover-risk-comparison');
    if (aiTakeoverEl) {
        aiTakeoverEl.innerHTML = `${formatPercent(aiTakeoverNoSlowdown)} &rarr; ${formatPercent(aiTakeoverSlowdown)}`;
    }

    // Update human power grab dashboard element
    const humanPowerGrabEl = document.getElementById('human-power-grab-risk-comparison');
    if (humanPowerGrabEl) {
        humanPowerGrabEl.innerHTML = `${formatPercent(humanPowerGrabNoSlowdown)} &rarr; ${formatPercent(humanPowerGrabSlowdown)}`;
    }
}

/**
 * Main function to update risk breakdown visualization
 */
function plotRiskBreakdown(data) {
    updateRiskBreakdownValues(data);
    updateRiskComparisonDashboard(data);
}

// Legacy function name for compatibility
function plotPCatastropheOverTime(data) {
    plotRiskBreakdown(data);
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { plotPCatastropheOverTime, plotRiskBreakdown, updateRiskBreakdownValues, updateRiskComparisonDashboard };
}
