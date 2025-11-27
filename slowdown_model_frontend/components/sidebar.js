// Sidebar Component JavaScript

// Collect all sidebar parameters and return as query string
function collectSidebarParams() {
    const params = new URLSearchParams();

    // Monte Carlo samples
    params.set('num_mc_samples', document.getElementById('num_mc_samples').value || '3');

    // P(AI Takeover) parameters
    params.set('p_ai_takeover_1_month', document.getElementById('p_ai_takeover_1_month').value || '0.15');
    params.set('p_ai_takeover_1_year', document.getElementById('p_ai_takeover_1_year').value || '0.10');
    params.set('p_ai_takeover_10_years', document.getElementById('p_ai_takeover_10_years').value || '0.05');

    // P(Human Power Grabs) parameters
    params.set('p_human_power_grabs_1_month', document.getElementById('p_human_power_grabs_1_month').value || '0.40');
    params.set('p_human_power_grabs_1_year', document.getElementById('p_human_power_grabs_1_year').value || '0.20');
    params.set('p_human_power_grabs_10_years', document.getElementById('p_human_power_grabs_10_years').value || '0.10');

    // Software proliferation parameters
    const weightStealing = document.getElementById('weight_stealing_enabled').value;
    params.set('weight_stealing', weightStealing);

    const algorithmStealing = document.getElementById('algorithm_stealing_enabled').value;
    params.set('algorithm_stealing', algorithmStealing);

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
