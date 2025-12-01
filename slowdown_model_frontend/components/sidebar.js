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
    // Element IDs map to backend default_parameters keys
    const inputMappings = {
        'monte_carlo_samples': defaults.monte_carlo_samples,
        'research_relevance_of_pre_handoff_discount': defaults.research_relevance_of_pre_handoff_discount,
        'increase_in_alignment_research_effort_during_slowdown': defaults.increase_in_alignment_research_effort_during_slowdown,
        'alignment_tax_after_handoff': defaults.alignment_tax_after_handoff,
        'safety_speedup_multiplier': defaults.safety_speedup_multiplier,
        'max_alignment_speedup_before_handoff': defaults.max_alignment_speedup_before_handoff,
        'p_misalignment_at_handoff_t1': defaults.p_misalignment_at_handoff_t1,
        'p_misalignment_at_handoff_t2': defaults.p_misalignment_at_handoff_t2,
        'p_misalignment_at_handoff_t3': defaults.p_misalignment_at_handoff_t3,
        'p_misalignment_after_handoff_t1': defaults.p_misalignment_after_handoff_t1,
        'p_misalignment_after_handoff_t2': defaults.p_misalignment_after_handoff_t2,
        'p_misalignment_after_handoff_t3': defaults.p_misalignment_after_handoff_t3,
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
        'weight_stealing_times': defaults.weight_stealing_times,
        'stealing_algorithms_up_to': defaults.stealing_algorithms_up_to,
    };

    for (const [selectId, value] of Object.entries(selectMappings)) {
        const el = document.getElementById(selectId);
        if (el && value !== undefined) {
            el.value = value;
        }
    }
}

// Collect all sidebar parameters and return as query string
// Only includes parameters that have values - backend uses its defaults for missing params
// Parameter names use dot notation to match backend SlowdownPageParameters structure
function collectSidebarParams() {
    const params = new URLSearchParams();

    // Helper to add param only if it has a value
    function addIfPresent(paramName, elementId) {
        const value = getElementValue(elementId, '');
        if (value !== '') {
            params.set(paramName, value);
        }
    }

    // Simulation settings (top-level SlowdownPageParameters attributes)
    addIfPresent('monte_carlo_samples', 'monte_carlo_samples');

    // PCatastrophe_parameters nested attributes (use dot notation)
    addIfPresent('PCatastrophe_parameters.research_relevance_of_pre_handoff_discount', 'research_relevance_of_pre_handoff_discount');
    addIfPresent('PCatastrophe_parameters.increase_in_alignment_research_effort_during_slowdown', 'increase_in_alignment_research_effort_during_slowdown');
    addIfPresent('PCatastrophe_parameters.alignment_tax_after_handoff', 'alignment_tax_after_handoff');
    addIfPresent('PCatastrophe_parameters.safety_speedup_multiplier', 'safety_speedup_multiplier');
    addIfPresent('PCatastrophe_parameters.max_alignment_speedup_before_handoff', 'max_alignment_speedup_before_handoff');

    // P(Misalignment at Handoff) parameters (previously p_ai_takeover)
    addIfPresent('PCatastrophe_parameters.p_misalignment_at_handoff_t1', 'p_misalignment_at_handoff_t1');
    addIfPresent('PCatastrophe_parameters.p_misalignment_at_handoff_t2', 'p_misalignment_at_handoff_t2');
    addIfPresent('PCatastrophe_parameters.p_misalignment_at_handoff_t3', 'p_misalignment_at_handoff_t3');

    // P(Misalignment after Handoff) parameters
    addIfPresent('PCatastrophe_parameters.p_misalignment_after_handoff_t1', 'p_misalignment_after_handoff_t1');
    addIfPresent('PCatastrophe_parameters.p_misalignment_after_handoff_t2', 'p_misalignment_after_handoff_t2');
    addIfPresent('PCatastrophe_parameters.p_misalignment_after_handoff_t3', 'p_misalignment_after_handoff_t3');

    // P(Human Power Grabs) parameters
    addIfPresent('PCatastrophe_parameters.p_human_power_grabs_t1', 'p_human_power_grabs_t1');
    addIfPresent('PCatastrophe_parameters.p_human_power_grabs_t2', 'p_human_power_grabs_t2');
    addIfPresent('PCatastrophe_parameters.p_human_power_grabs_t3', 'p_human_power_grabs_t3');

    // Software proliferation parameters (use dot notation)
    addIfPresent('software_proliferation.weight_stealing_times', 'weight_stealing_times');
    addIfPresent('software_proliferation.stealing_algorithms_up_to', 'stealing_algorithms_up_to');

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
