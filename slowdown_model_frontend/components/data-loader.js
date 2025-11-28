// Data Loader Module - Handles all data fetching for the slowdown model

// Global variable to store the slowdown model data
let slowdownModelData = null;

// Get the current slowdown model data
function getSlowdownModelData() {
    return slowdownModelData;
}

// Set the slowdown model data (useful for external updates)
function setSlowdownModelData(data) {
    slowdownModelData = data;
}

// Load fast trajectory data (deterministic, 4 curves in parallel)
// Returns immediately with median-only data for the AI R&D Speedup Trajectory plot
async function loadTrajectoryDataFast(options = {}) {
    const {
        onComplete = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        const queryParams = collectParams ? collectParams() : '';

        // Show loading indicator for trajectory plot
        if (typeof showTakeoffLoadingIndicator === 'function') {
            showTakeoffLoadingIndicator();
        }

        const response = await fetch(`/get_trajectory_data_fast?${queryParams}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Store the trajectory data
        slowdownModelData = data;

        if (onComplete) {
            onComplete(data);
        }

        return data;
    } catch (error) {
        console.error('Error loading trajectory data:', error);
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError(error.message || 'Error loading trajectory data.');
        }
        throw error;
    }
}

// Load uncertainty data with streaming progress (MC simulations for covert and proxy_project)
// This is slower and independent of the trajectory plot
async function loadUncertaintyDataStream(options = {}) {
    const {
        onProgress = null,
        onComplete = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        const queryParams = collectParams ? collectParams() : '';

        // Show loading indicator for uncertainty plot
        if (typeof showCovertUncertaintyLoadingIndicator === 'function') {
            showCovertUncertaintyLoadingIndicator();
        }

        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(`/get_uncertainty_data_stream?${queryParams}`);

            eventSource.onmessage = function(event) {
                const msg = JSON.parse(event.data);

                if (msg.type === 'progress') {
                    if (onProgress) {
                        onProgress(msg.current, msg.total, msg.trajectory);
                    }
                    if (typeof updateCovertUncertaintyProgress === 'function') {
                        updateCovertUncertaintyProgress(msg.current, msg.total, msg.trajectory);
                    }
                } else if (msg.type === 'complete') {
                    eventSource.close();

                    // Merge uncertainty data into the main slowdownModelData
                    if (slowdownModelData && msg.data && msg.data.monte_carlo) {
                        // Update only the MC data for covert and proxy_project
                        if (!slowdownModelData.monte_carlo) {
                            slowdownModelData.monte_carlo = {};
                        }
                        if (msg.data.monte_carlo.covert) {
                            slowdownModelData.monte_carlo.covert = msg.data.monte_carlo.covert;
                        }
                        if (msg.data.monte_carlo.proxy_project) {
                            slowdownModelData.monte_carlo.proxy_project = msg.data.monte_carlo.proxy_project;
                        }
                    }

                    if (onComplete) {
                        onComplete(msg.data);
                    }
                    resolve(msg.data);
                } else if (msg.type === 'error') {
                    eventSource.close();
                    console.error('Server error:', msg.error);
                    if (onError) {
                        onError(msg.error);
                    }
                    if (typeof showCovertUncertaintyError === 'function') {
                        showCovertUncertaintyError(msg.error);
                    }
                    reject(new Error(msg.error));
                } else if (msg.type === 'keepalive') {
                    // Ignore keepalive messages
                }
            };

            eventSource.onerror = function(error) {
                eventSource.close();
                console.error('EventSource error:', error);
                if (onError) {
                    onError('Connection error');
                }
                if (typeof showCovertUncertaintyError === 'function') {
                    showCovertUncertaintyError('Connection error while loading uncertainty data.');
                }
                reject(new Error('EventSource connection error'));
            };
        });
    } catch (error) {
        console.error('Error loading uncertainty data:', error);
        if (onError) {
            onError(error.message);
        }
        throw error;
    }
}

// Main entry point: Load both trajectory and uncertainty data in parallel
// Trajectory data loads fast and renders immediately
// Uncertainty data loads separately with progress updates
async function loadAllSlowdownData(options = {}) {
    const {
        onProgress = null,
        onComplete = null,
        onError = null,
        collectParams = null,
        onTrajectoryComplete = null,
        onUncertaintyComplete = null
    } = options;

    try {
        // Show loading indicators for both plots
        if (typeof showTakeoffLoadingIndicator === 'function') {
            showTakeoffLoadingIndicator();
        }
        if (typeof showCovertUncertaintyLoadingIndicator === 'function') {
            showCovertUncertaintyLoadingIndicator();
        }

        // Start both requests in parallel
        const trajectoryPromise = loadTrajectoryDataFast({
            collectParams,
            onComplete: (data) => {
                console.log('Trajectory data loaded, rendering plots...');
                // Render the trajectory plot immediately
                if (typeof plotTakeoffModel === 'function') {
                    plotTakeoffModel(data);
                }
                // Render the compute plot
                if (typeof plotCovertCompute === 'function') {
                    plotCovertCompute(data);
                }
                // Render P(catastrophe) plots
                if (typeof plotPCatastropheFromData === 'function') {
                    plotPCatastropheFromData(data);
                }
                // Render P(catastrophe) over time plot
                if (typeof plotPCatastropheOverTime === 'function') {
                    plotPCatastropheOverTime(data);
                }
                // Render optimal compute cap over time plot
                if (typeof plotOptimalComputeCapOverTime === 'function') {
                    plotOptimalComputeCapOverTime(data);
                }
                if (onTrajectoryComplete) {
                    onTrajectoryComplete(data);
                }
            },
            onError: (err) => {
                console.error('Trajectory data error:', err);
                if (onError) {
                    onError(err);
                }
            }
        });

        const uncertaintyPromise = loadUncertaintyDataStream({
            collectParams,
            onProgress,
            onComplete: (data) => {
                console.log('Uncertainty data loaded, rendering uncertainty plot...');
                // Render the uncertainty plot with MC data
                if (typeof plotCovertUncertainty === 'function') {
                    plotCovertUncertainty(slowdownModelData);
                }
                if (onUncertaintyComplete) {
                    onUncertaintyComplete(data);
                }
            },
            onError: (err) => {
                console.error('Uncertainty data error:', err);
                // Don't call onError here - trajectory data is more important
                // Just log the error and show it in the uncertainty plot
            }
        });

        // Wait for both to complete
        const [trajectoryData, uncertaintyData] = await Promise.all([
            trajectoryPromise,
            uncertaintyPromise.catch(err => {
                // Don't fail the whole thing if uncertainty fails
                console.error('Uncertainty loading failed:', err);
                return null;
            })
        ]);

        // Call overall onComplete
        if (onComplete) {
            onComplete(slowdownModelData);
        }

        return slowdownModelData;
    } catch (error) {
        console.error('Error loading slowdown model data:', error);
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError(error.message || 'Error loading data. Please refresh the page.');
        }
        throw error;
    }
}

// Legacy: Fetch all data from backend using the old streaming endpoint (deprecated)
async function loadAllSlowdownDataLegacy(options = {}) {
    const {
        onProgress = null,
        onComplete = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        // Collect all parameters from sidebar
        const queryParams = collectParams ? collectParams() : '';

        // Show loading indicators
        if (typeof showTakeoffLoadingIndicator === 'function') {
            showTakeoffLoadingIndicator();
        }
        if (typeof showCovertUncertaintyLoadingIndicator === 'function') {
            showCovertUncertaintyLoadingIndicator();
        }

        // Use Server-Sent Events for streaming progress
        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(`/get_slowdown_model_data_stream?${queryParams}`);

            eventSource.onmessage = function(event) {
                console.log('SSE message received, type:', event.data.substring(0, 50));
                const msg = JSON.parse(event.data);

                if (msg.type === 'progress') {
                    // Update progress bars
                    if (onProgress) {
                        onProgress(msg.current, msg.total, msg.trajectory);
                    }
                    if (typeof updateTakeoffProgress === 'function') {
                        updateTakeoffProgress(msg.current, msg.total, msg.trajectory);
                    }
                    if (typeof updateCovertUncertaintyProgress === 'function') {
                        updateCovertUncertaintyProgress(msg.current, msg.total, msg.trajectory);
                    }
                } else if (msg.type === 'complete') {
                    // Data received - close connection and process
                    console.log('Complete message received, calling onComplete...');
                    eventSource.close();
                    slowdownModelData = msg.data;

                    if (onComplete) {
                        onComplete(slowdownModelData);
                        console.log('onComplete finished');
                    }
                    resolve(slowdownModelData);
                } else if (msg.type === 'error') {
                    eventSource.close();
                    console.error('Server error:', msg.error);
                    if (onError) {
                        onError(msg.error);
                    }
                    if (typeof showTakeoffError === 'function') {
                        showTakeoffError(msg.error);
                    }
                    reject(new Error(msg.error));
                } else if (msg.type === 'status') {
                    // Update status text
                    if (typeof updateTakeoffStatus === 'function') {
                        updateTakeoffStatus(msg.message);
                    }
                } else if (msg.type === 'keepalive') {
                    // Ignore keepalive messages
                }
            };

            eventSource.onerror = function(error) {
                eventSource.close();
                console.error('EventSource error:', error);
                // Fallback to non-streaming endpoint
                console.log('Falling back to non-streaming endpoint...');
                loadAllSlowdownDataNonStreaming(options).then(resolve).catch(reject);
            };
        });
    } catch (error) {
        console.error('Error loading slowdown model data:', error);
        // Log error to server for debugging
        try {
            fetch('/log_client_error', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: 'Error loading slowdown model data: ' + error.message,
                    stack: error.stack
                })
            });
        } catch (logError) {
            // Ignore logging errors
        }
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError(error.message || 'Error loading data. Please refresh the page.');
        }
        if (typeof showComputePlotError === 'function') {
            showComputePlotError(error.message || 'Error loading data. Please refresh the page.');
        }
        throw error;
    }
}

// Non-streaming fallback for browsers that don't support SSE
async function loadAllSlowdownDataNonStreaming(options = {}) {
    const {
        onComplete = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        const plotContainer = document.getElementById('slowdownPlot');
        if (plotContainer) {
            plotContainer.innerHTML = '<p style="text-align: center; color: #888; padding: 100px;">Loading data... (this may take a minute)</p>';
        }

        const queryParams = collectParams ? collectParams() : '';
        const response = await fetch(`/get_slowdown_model_data?${queryParams}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        slowdownModelData = await response.json();

        if (onComplete) {
            onComplete(slowdownModelData);
        }

        return slowdownModelData;
    } catch (error) {
        console.error('Error loading slowdown model data:', error);
        // Log error to server for debugging
        try {
            fetch('/log_client_error', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: 'Error loading slowdown model data: ' + error.message,
                    stack: error.stack
                })
            });
        } catch (logError) {
            // Ignore logging errors
        }
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError(error.message || 'Error loading data. Please refresh the page.');
        }
        if (typeof showComputePlotError === 'function') {
            showComputePlotError(error.message || 'Error loading data. Please refresh the page.');
        }
        throw error;
    }
}

// Legacy alias for backwards compatibility
async function loadAndPlotTakeoffModel(options = {}) {
    return await loadAllSlowdownData(options);
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        getSlowdownModelData,
        setSlowdownModelData,
        loadTrajectoryDataFast,
        loadUncertaintyDataStream,
        loadAllSlowdownData,
        loadAllSlowdownDataLegacy,
        loadAllSlowdownDataNonStreaming,
        loadAndPlotTakeoffModel
    };
}
