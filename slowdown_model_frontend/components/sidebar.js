// Sidebar Component JavaScript

// Helper to safely get element value with fallback
function getElementValue(id, fallback) {
    const el = document.getElementById(id);
    return el ? (el.value || fallback) : fallback;
}

// Populate sidebar inputs from backend defaults
function populateSidebarDefaults(data) {
    const defaults = data.default_parameters;
    if (!defaults) {
        console.warn('No default_parameters in data');
        return;
    }

    // Set each input value from defaults
    const inputMappings = {
        'num_mc_samples': defaults.num_mc_samples,
        'handoff_speedup_threshold': defaults.handoff_speedup_threshold,
        'research_relevance_of_pre_handoff_discount': defaults.research_relevance_of_pre_handoff_discount,
        'increase_in_alignment_research_effort_during_slowdown': defaults.increase_in_alignment_research_effort_during_slowdown,
        'alignment_tax_after_handoff_relative_to_during_handoff': defaults.alignment_tax_after_handoff_relative_to_during_handoff,
        'safety_speedup_exponent': defaults.safety_speedup_exponent,
        'p_ai_takeover_t1': defaults.p_ai_takeover_t1,
        'p_ai_takeover_t2': defaults.p_ai_takeover_t2,
        'p_ai_takeover_t3': defaults.p_ai_takeover_t3,
        'p_human_power_grabs_t1': defaults.p_human_power_grabs_t1,
        'p_human_power_grabs_t2': defaults.p_human_power_grabs_t2,
        'p_human_power_grabs_t3': defaults.p_human_power_grabs_t3,
    };

    for (const [inputId, value] of Object.entries(inputMappings)) {
        const el = document.getElementById(inputId);
        if (el && value !== undefined) {
            el.value = value;
        }
    }

    // Set select elements
    const selectMappings = {
        'weight_stealing_enabled': defaults.weight_stealing_enabled,
        'algorithm_stealing_enabled': defaults.algorithm_stealing_enabled,
    };

    for (const [selectId, value] of Object.entries(selectMappings)) {
        const el = document.getElementById(selectId);
        if (el && value !== undefined) {
            el.value = value;
        }
    }
}

// Collect all sidebar parameters and return as query string
function collectSidebarParams() {
    const params = new URLSearchParams();

    // Simulation settings
    params.set('num_mc_samples', getElementValue('num_mc_samples', '1'));
    params.set('handoff_speedup_threshold', getElementValue('handoff_speedup_threshold', '20'));
    params.set('research_relevance_of_pre_handoff_discount', getElementValue('research_relevance_of_pre_handoff_discount', '0.1'));
    params.set('increase_in_alignment_research_effort_during_slowdown', getElementValue('increase_in_alignment_research_effort_during_slowdown', '1.5'));
    params.set('alignment_tax_after_handoff_relative_to_during_handoff', getElementValue('alignment_tax_after_handoff_relative_to_during_handoff', '2.0'));

    // Safety research speedup exponent
    params.set('safety_speedup_exponent', getElementValue('safety_speedup_exponent', '0.5'));

    // P(AI Takeover) parameters
    params.set('p_ai_takeover_t1', getElementValue('p_ai_takeover_t1', '0.40'));
    params.set('p_ai_takeover_t2', getElementValue('p_ai_takeover_t2', '0.15'));
    params.set('p_ai_takeover_t3', getElementValue('p_ai_takeover_t3', '0.05'));

    // P(Human Power Grabs) parameters
    params.set('p_human_power_grabs_t1', getElementValue('p_human_power_grabs_t1', '0.40'));
    params.set('p_human_power_grabs_t2', getElementValue('p_human_power_grabs_t2', '0.15'));
    params.set('p_human_power_grabs_t3', getElementValue('p_human_power_grabs_t3', '0.05'));

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
    module.exports = { collectSidebarParams, initSidebar, populateSidebarDefaults };
}
