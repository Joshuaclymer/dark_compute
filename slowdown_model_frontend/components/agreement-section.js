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
    module.exports = { loadAgreementSection, updateAgreementSection };
}
