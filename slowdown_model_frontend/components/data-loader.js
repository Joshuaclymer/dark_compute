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

// Main entry point: Load all slowdown data using streaming endpoint
async function loadAllSlowdownData(options = {}) {
    const {
        onProgress = null,
        onComplete = null,
        onPartial = null,
        onError = null,
        collectParams = null
    } = options;

    try {
        const queryParams = collectParams ? collectParams() : '';

        // Show loading indicators
        if (typeof showTakeoffLoadingIndicator === 'function') {
            showTakeoffLoadingIndicator();
        }
        if (typeof showCovertUncertaintyLoadingIndicator === 'function') {
            showCovertUncertaintyLoadingIndicator();
        }
        if (typeof showAgreementRiskLoadingIndicator === 'function') {
            showAgreementRiskLoadingIndicator();
        }

        // Use Server-Sent Events for streaming progress
        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(`/get_slowdown_model_data_stream?${queryParams}`);

            eventSource.onmessage = function(event) {
                const msg = JSON.parse(event.data);

                if (msg.type === 'progress') {
                    if (onProgress) {
                        onProgress(msg.current, msg.total, msg.trajectory);
                    }
                    if (typeof updateTakeoffProgress === 'function') {
                        updateTakeoffProgress(msg.current, msg.total, msg.trajectory);
                    }
                } else if (msg.type === 'partial') {
                    // Partial data available - render immediately for fast feedback
                    slowdownModelData = msg.data;
                    if (onPartial) {
                        onPartial(slowdownModelData);
                    }
                    // Also call onComplete with partial data so plots render quickly
                    if (onComplete) {
                        onComplete(slowdownModelData);
                    }
                } else if (msg.type === 'complete') {
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
                } else if (msg.type === 'status') {
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
                if (onError) {
                    onError('Connection error');
                }
                if (typeof showTakeoffError === 'function') {
                    showTakeoffError('Connection error. Please refresh the page.');
                }
                reject(new Error('EventSource connection error'));
            };
        });
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
        loadAndPlotTakeoffModel
    };
}
