// Agreement Section Component JavaScript

// Load the agreement section HTML
async function loadAgreementSection() {
    const container = document.getElementById('agreement-section-container');
    if (!container) return;

    try {
        const response = await fetch('/slowdown_model_frontend/components/agreement-section.html');
        const html = await response.text();
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading agreement section:', error);
    }
}

// Plot compute over time in agreement section (styled to match takeoff trajectory plot)
function plotAgreementComputeOverTime(data) {
    const combinedData = data.combined_covert_compute;
    const largestCompanyData = data.largest_company_compute;
    const prcNoSlowdownData = data.prc_no_slowdown_compute;
    const proxyProjectData = data.proxy_project_compute;
    const usSlowdownData = data.us_slowdown_compute;  // Effective compute for US slowdown

    const container = document.getElementById('agreementComputePlot');
    if (!container) return;

    if (!combinedData || !combinedData.years || !combinedData.median) {
        container.innerHTML = '<p style="text-align: center; color: #888; padding: 20px;">No compute data available</p>';
        return;
    }

    const years = combinedData.years;
    const median = combinedData.median;
    const agreement_year = data.agreement_year;

    // Color scheme matching takeoff trajectory plot
    const usColor = '#5B8DBE';    // Blue for US
    const prcColor = '#C77CAA';   // Purple/pink for PRC

    const traces = [];

    // 1. US no slowdown (dashed) - Largest U.S. Company
    if (largestCompanyData && largestCompanyData.years && largestCompanyData.compute) {
        traces.push({
            x: largestCompanyData.years,
            y: largestCompanyData.compute,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: usColor,
                width: 3,
                dash: 'dash'
            },
            name: 'US (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 1.5. US with slowdown (solid) - Effective compute during AI slowdown
    if (usSlowdownData && usSlowdownData.years && usSlowdownData.compute) {
        // Plot the full trajectory (no filtering)
        traces.push({
            x: usSlowdownData.years,
            y: usSlowdownData.compute,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: usColor,
                width: 3
            },
            name: 'US (with slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 2. PRC no slowdown (dashed)
    if (prcNoSlowdownData && prcNoSlowdownData.years && prcNoSlowdownData.median) {
        traces.push({
            x: prcNoSlowdownData.years,
            y: prcNoSlowdownData.median,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: prcColor,
                width: 3,
                dash: 'dash'
            },
            name: 'PRC (no slowdown)',
            hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
        });
    }

    // 3. Proxy Project (dotted black) - only from agreement year onwards
    if (proxyProjectData && proxyProjectData.years && proxyProjectData.compute) {
        // Filter to only include years from agreement year onwards
        const proxyYears = [];
        const proxyCompute = [];
        for (let i = 0; i < proxyProjectData.years.length; i++) {
            if (proxyProjectData.years[i] >= agreement_year) {
                proxyYears.push(proxyProjectData.years[i]);
                proxyCompute.push(proxyProjectData.compute[i]);
            }
        }

        if (proxyYears.length > 0) {
            traces.push({
                x: proxyYears,
                y: proxyCompute,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#000000',
                    width: 3,
                    dash: 'dot'
                },
                name: 'Proxy Project',
                hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
            });
        }
    }

    // 4. PRC covert compute (solid)
    traces.push({
        x: years,
        y: median,
        type: 'scatter',
        mode: 'lines',
        line: {
            color: prcColor,
            width: 3
        },
        name: 'PRC (with slowdown)',
        hovertemplate: 'Year: %{x:.1f}<br>H100e: %{y:,.0f}<extra></extra>'
    });

    // Add a dummy trace for the agreement line legend entry
    if (agreement_year) {
        traces.push({
            x: [null],
            y: [null],
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#888',
                width: 2,
                dash: 'dot'
            },
            name: 'Agreement start',
            showlegend: true
        });
    }

    // Determine end year from the data
    const endYear = years[years.length - 1];

    const layout = {
        xaxis: {
            title: 'Year',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            automargin: true,
            range: [2026, endYear]
        },
        yaxis: {
            title: 'H100-equivalents',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            type: 'log',
            automargin: true
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1
        },
        hovermode: 'closest',
        margin: { l: 50, r: 20, t: 10, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        shapes: agreement_year ? [{
            type: 'line',
            x0: agreement_year,
            x1: agreement_year,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: '#888',
                width: 2,
                dash: 'dot'
            }
        }] : []
    };

    // Clear container before plotting
    container.innerHTML = '';

    Plotly.newPlot('agreementComputePlot', traces, layout, {displayModeBar: false, responsive: true});
}

// Update the agreement section with data
function updateAgreementSection(data) {
    // Update speedup value in title
    const speedupValue = document.getElementById('agreement-speedup-value');
    if (speedupValue && data.speedup) {
        speedupValue.textContent = `${data.speedup}x AI R&D`;
    }

    // Update outcome values
    if (data.outcomes) {
        const slowdownEl = document.getElementById('outcome-slowdown');
        if (slowdownEl && data.outcomes.slowdown !== undefined) {
            slowdownEl.textContent = data.outcomes.slowdown;
        }

        const takeoverEl = document.getElementById('outcome-takeover-risk');
        if (takeoverEl && data.outcomes.takeoverRisk !== undefined) {
            takeoverEl.innerHTML = data.outcomes.takeoverRisk;
        }

        const prcEl = document.getElementById('outcome-prc-risk');
        if (prcEl && data.outcomes.prcRisk !== undefined) {
            prcEl.innerHTML = data.outcomes.prcRisk;
        }

        const computeReductionEl = document.getElementById('outcome-compute-reduction');
        if (computeReductionEl && data.outcomes.computeReduction !== undefined) {
            computeReductionEl.textContent = data.outcomes.computeReduction;
        }

        const researchReductionEl = document.getElementById('outcome-research-reduction');
        if (researchReductionEl && data.outcomes.researchReduction !== undefined) {
            researchReductionEl.textContent = data.outcomes.researchReduction;
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', loadAgreementSection);

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { loadAgreementSection, updateAgreementSection, plotAgreementComputeOverTime };
}
