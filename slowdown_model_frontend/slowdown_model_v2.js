// Slowdown Model v2 - Main Application Script
// This file assembles all components and initializes the application

// Load HTML components into their containers
async function loadComponents() {
    const components = [
        { id: 'sidebar-container', path: '/slowdown_model_frontend/components/sidebar.html' },
        { id: 'header-container', path: '/slowdown_model_frontend/components/header.html' },
        { id: 'takeoff-trajectory-container', path: '/slowdown_model_frontend/components/takeoff-trajectory-plot.html' },
        { id: 'p-catastrophe-dashboard-container', path: '/slowdown_model_frontend/components/p-catastrophe-dashboard.html' },
        { id: 'compute-plot-container', path: '/slowdown_model_frontend/components/compute-plot.html' },
        { id: 'p-catastrophe-plots-container', path: '/slowdown_model_frontend/components/p-catastrophe-plots.html' },
        { id: 'explanation-container', path: '/slowdown_model_frontend/components/explanation.html' }
    ];

    // Load all components in parallel
    const loadPromises = components.map(async (component) => {
        try {
            const response = await fetch(component.path);
            if (!response.ok) {
                throw new Error(`Failed to load ${component.path}: ${response.status}`);
            }
            const html = await response.text();
            const container = document.getElementById(component.id);
            if (container) {
                container.innerHTML = html;
            }
        } catch (error) {
            console.error(`Error loading component ${component.id}:`, error);
        }
    });

    await Promise.all(loadPromises);
}

// Plot all charts from the data
function plotAllCharts(data) {
    plotTakeoffModel(data);
    plotCovertCompute(data);
    plotPCatastropheFromData(data);
    renderPCatastropheDashboard(data);
}

// Update all plots by reloading data from backend
function updateAllPlots() {
    loadAllSlowdownData({
        collectParams: collectSidebarParams,
        onComplete: plotAllCharts,
        onError: (error) => {
            console.error('Error updating plots:', error);
        }
    });
}

// Legacy alias
function updatePCatastrophePlots() {
    updateAllPlots();
}

// Initialize the application
async function initApp() {
    // First, load all HTML components
    await loadComponents();

    // Initialize sidebar with update callback
    initSidebar(updateAllPlots);

    // Load initial data
    loadAllSlowdownData({
        collectParams: collectSidebarParams,
        onComplete: plotAllCharts,
        onError: (error) => {
            console.error('Error loading initial data:', error);
        }
    });
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', initApp);
