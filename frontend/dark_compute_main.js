// JavaScript for Dark Compute Model - Main Coordinator
// This file coordinates all three subsections: Top, Rate, and Detection

function plotDarkComputeModel(data) {
    // Plot all sections in order

    // 1. Plot top section (dashboard and main plots)
    // plotH100YearsTimeSeries(data);  // Commented out - replaced with H100-years CCDF
    plotTimeToDetectionCcdf(data);
    plotChipProductionReductionCcdf(data);
    plotAiRdReductionCcdf(data);
    plotProjectH100YearsCcdf(data);
    updateDarkComputeModelDashboard(data);

    // 2. Plot rate of computation section
    plotDarkComputeRateSection(data);

    // 3. Plot detection likelihood section
    plotDarkComputeDetectionSection(data);
}
