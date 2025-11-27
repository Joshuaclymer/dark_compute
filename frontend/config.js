// Global configuration for detection thresholds
// Update these values to change all plots automatically
const DETECTION_CONFIG = {
    // Likelihood ratio thresholds for detection
    LIKELIHOOD_RATIOS: [1, 2, 4],

    // Colors for each threshold (Purple, Blue, Blue-green)
    COLORS: ['#9B7BB3', '#5B8DBE', '#5AA89B'],

    // The highest threshold used for dashboard values and primary detection
    PRIMARY_THRESHOLD: 4,

    // Get threshold config as array of objects
    getThresholds: function() {
        return this.LIKELIHOOD_RATIOS.map((lr, index) => ({
            value: lr,
            multiplier: lr,
            color: this.COLORS[index % this.COLORS.length],
            label: `${lr}x update`
        }));
    }
};
