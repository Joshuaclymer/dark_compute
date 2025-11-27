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

// Fetch all data from backend and return via callbacks (with streaming progress updates)
async function loadAllSlowdownData(options = {}) {
    const {
        onProgress = null,
        onComplete = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        // Collect all parameters from sidebar
        const queryParams = collectParams ? collectParams() : '';

        // Show loading indicator
        if (typeof showTakeoffLoadingIndicator === 'function') {
            showTakeoffLoadingIndicator();
        }

        // Use Server-Sent Events for streaming progress
        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(`/get_slowdown_model_data_stream?${queryParams}`);

            eventSource.onmessage = function(event) {
                const msg = JSON.parse(event.data);

                if (msg.type === 'progress') {
                    // Update progress bar
                    if (onProgress) {
                        onProgress(msg.current, msg.total, msg.trajectory);
                    }
                    if (typeof updateTakeoffProgress === 'function') {
                        updateTakeoffProgress(msg.current, msg.total, msg.trajectory);
                    }
                } else if (msg.type === 'complete') {
                    // Data received - close connection and process
                    eventSource.close();
                    slowdownModelData = msg.data;

                    if (onComplete) {
                        onComplete(slowdownModelData);
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
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError('Error loading data. Please refresh the page.');
        }
        if (typeof showComputePlotError === 'function') {
            showComputePlotError('Error loading data. Please refresh the page.');
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
        if (onError) {
            onError(error.message);
        }
        if (typeof showTakeoffError === 'function') {
            showTakeoffError('Error loading data. Please refresh the page.');
        }
        if (typeof showComputePlotError === 'function') {
            showComputePlotError('Error loading data. Please refresh the page.');
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
        loadAllSlowdownData,
        loadAllSlowdownDataNonStreaming,
        loadAndPlotTakeoffModel
    };
}
