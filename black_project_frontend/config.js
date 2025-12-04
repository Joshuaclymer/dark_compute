// Global color palette for consistent styling across all plots
// Update these values to change colors throughout the application
const COLOR_PALETTE = {
    // Primary colors for data series (named by usage)
    chip_stock: '#5E6FB8',           // Initial chip stock, PRC chip stock evidence (Indigo)
    fab: '#E9A842',                  // Covert fab production, SME stock evidence (Marigold)
    datacenters_and_energy: '#4AA896', // Satellite/datacenter evidence (Viridian)
    detection: '#7BA3C4',            // Detection/LR plots, energy consumption (Pewter Blue)

    // Secondary/accent colors
    survival_rate: '#E05A4F',        // Survival rate (Vermillion)
    gray: '#7F8C8D',                 // Neutral/disabled

    // Helper function to get rgba version with alpha
    rgba: function(colorName, alpha) {
        const hex = this[colorName];
        if (!hex) return `rgba(0,0,0,${alpha})`;
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },

    // Helper to get hex with alpha suffix (for Plotly fillcolor)
    hexAlpha: function(colorName, alphaHex) {
        return this[colorName] + alphaHex;
    },

    // Helper to darken a color by a factor (0-1, where 0.8 = 20% darker)
    darken: function(colorName, factor) {
        const hex = this[colorName];
        if (!hex) return '#000000';
        const r = Math.round(parseInt(hex.slice(1, 3), 16) * factor);
        const g = Math.round(parseInt(hex.slice(3, 5), 16) * factor);
        const b = Math.round(parseInt(hex.slice(5, 7), 16) * factor);
        return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
    },

    // Helper to lighten a color by a factor (1.2 = 20% lighter)
    lighten: function(colorName, factor) {
        const hex = this[colorName];
        if (!hex) return '#FFFFFF';
        const r = Math.min(255, Math.round(parseInt(hex.slice(1, 3), 16) * factor));
        const g = Math.min(255, Math.round(parseInt(hex.slice(3, 5), 16) * factor));
        const b = Math.min(255, Math.round(parseInt(hex.slice(5, 7), 16) * factor));
        return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
    },

    // Helper to convert any hex color to rgba (works with computed colors like darken/lighten results)
    hexToRgba: function(hex, alpha) {
        if (!hex || !hex.startsWith('#')) return `rgba(0,0,0,${alpha})`;
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
};

// Global configuration for detection thresholds
// Update these values to change all plots automatically
const DETECTION_CONFIG = {
    // Likelihood ratio thresholds for detection
    LIKELIHOOD_RATIOS: [1, 2, 4],

    // Colors for each threshold (chip_stock, detection, datacenters_and_energy)
    COLORS: [COLOR_PALETTE.chip_stock, COLOR_PALETTE.detection, COLOR_PALETTE.datacenters_and_energy],

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
