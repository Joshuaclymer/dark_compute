// Sidebar Component JavaScript

// Helper to safely get element value with fallback
function getElementValue(id, fallback) {
    const el = document.getElementById(id);
    return el ? (el.value || fallback) : fallback;
}

// Collect all sidebar parameters and return as query string
function collectSidebarParams() {
    const params = new URLSearchParams();

    // Monte Carlo samples
    params.set('num_mc_samples', getElementValue('num_mc_samples', '1'));

    // P(AI Takeover) parameters
    params.set('p_ai_takeover_1_month', getElementValue('p_ai_takeover_1_month', '0.15'));
    params.set('p_ai_takeover_1_year', getElementValue('p_ai_takeover_1_year', '0.10'));
    params.set('p_ai_takeover_10_years', getElementValue('p_ai_takeover_10_years', '0.05'));

    // P(Human Power Grabs) parameters
    params.set('p_human_power_grabs_1_month', getElementValue('p_human_power_grabs_1_month', '0.40'));
    params.set('p_human_power_grabs_1_year', getElementValue('p_human_power_grabs_1_year', '0.20'));
    params.set('p_human_power_grabs_10_years', getElementValue('p_human_power_grabs_10_years', '0.10'));

    // Software proliferation parameters
    params.set('weight_stealing', getElementValue('weight_stealing_enabled', 'SC'));
    params.set('algorithm_stealing', getElementValue('algorithm_stealing_enabled', 'SAR'));

    return params.toString();
}

// Initialize sidebar event listeners
function initSidebar(onUpdateCallback) {
    // Add event listener for update button
    const updateButton = document.getElementById('updatePlots');
    if (updateButton) {
        updateButton.addEventListener('click', function(e) {
            e.preventDefault();
            if (onUpdateCallback) {
                onUpdateCallback();
            }
        });
    }

    // Add Enter key listener to all number input fields in sidebar
    const sidebarInputs = document.querySelectorAll('.slowdown-sidebar input[type="number"]');
    sidebarInputs.forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (onUpdateCallback) {
                    onUpdateCallback();
                }
            }
        });
    });
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { collectSidebarParams, initSidebar };
}
