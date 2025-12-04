// Standalone function to create error bars plot for Intelligence Accuracy
function createIntelligenceAccuracyPlot() {
  const elementId = 'intelligenceAccuracyPlot';

  // Website color scheme - use palette colors
  const COLORS = {
    'purple': COLOR_PALETTE.chip_stock,      // Indigo #6E7FD9
    'blue': COLOR_PALETTE.detection,          // Pewter Blue #8DB8DA
    'teal': COLOR_PALETTE.datacenters_and_energy,  // Viridian #56C4AE
    'dark_teal': '#3D9E8A',                   // Darker viridian
    'red': COLOR_PALETTE.survival_rate,       // Vermillion #F0655A
    'purple_alt': COLOR_PALETTE.chip_stock,   // Indigo #6E7FD9
    'light_teal': '#7DD4C0'                   // Lighter viridian
  };

  // Helper functions
  function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  function linspace(start, end, num) {
    const arr = [];
    const step = (end - start) / (num - 1);
    for (let i = 0; i < num; i++) {
      arr.push(start + step * i);
    }
    return arr;
  }

  // Data for stated error bars (excluding Russian Federation nuclear warheads with min: 1000, max: 2000)
  const statedErrorBars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160},
    {"category": "Nuclear Warheads", "min": 140, "max": 157},
    {"category": "Nuclear Warheads", "min": 225, "max": 300},
    {"category": "Nuclear Warheads", "min": 60, "max": 80},
    {"category": "Fissile material (kg)", "min": 25, "max": 35},
    {"category": "Fissile material (kg)", "min": 30, "max": 50},
    {"category": "Fissile material (kg)", "min": 17, "max": 33},
    {"category": "Fissile material (kg)", "min": 335, "max": 400},
    {"category": "Fissile material (kg)", "min": 330, "max": 580},
    {"category": "Fissile material (kg)", "min": 240, "max": 395},
    {"category": "ICBM launchers", "min": 10, "max": 25},
    {"category": "ICBM launchers", "min": 10, "max": 25},
    {"category": "ICBM launchers", "min": 105, "max": 120},
    {"category": "ICBM launchers", "min": 200, "max": 240},
    {"category": "Intercontinental missiles", "min": 180, "max": 190},
    {"category": "Intercontinental missiles", "min": 200, "max": 300},
    {"category": "Intercontinental missiles", "min": 192, "max": 192}
  ];

  // Calculate central estimates and bounds
  const centralEstimates = [];
  const lowerBounds = [];
  const upperBounds = [];
  const categories = [];
  const upperPercentErrors = [];
  const lowerPercentErrors = [];

  statedErrorBars.forEach(entry => {
    const central = (entry.min + entry.max) / 2;
    centralEstimates.push(central);
    lowerBounds.push(entry.min);
    upperBounds.push(entry.max);
    categories.push(entry.category);

    const upperError = ((entry.max - central) / central) * 100;
    const lowerError = ((central - entry.min) / central) * 100;
    upperPercentErrors.push(upperError);
    lowerPercentErrors.push(lowerError);
  });

  // Calculate median percent errors
  const medianUpperError = median(upperPercentErrors);
  const medianLowerError = median(lowerPercentErrors);

  // Calculate slopes
  const upperSlope = 1 + (medianUpperError / 100);
  const lowerSlope = 1 - (medianLowerError / 100);

  // Data for estimate vs reality
  const estimates = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208];
  const groundTruths = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287];
  const estimateCategories = [
    "Aircraft", "Aircraft", "Aircraft",
    "Chemical Weapons (metric tons)", "Chemical Weapons (metric tons)",
    "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers",
    "Nuclear Warheads (/10)", "Nuclear Warheads (/10)",
    "Ground combat systems (/10)", "Ground combat systems (/10)", "Ground combat systems (/10)",
    "Troops (/1000)"
  ];

  // Calculate median estimate error
  const estimatePercentErrors = [];
  for (let i = 0; i < estimates.length; i++) {
    if (groundTruths[i] !== 0) {
      estimatePercentErrors.push(Math.abs((estimates[i] - groundTruths[i]) / groundTruths[i]) * 100);
    }
  }
  const medianEstimateError = median(estimatePercentErrors);

  const estimateUpperSlope = 1 + (medianEstimateError / 100);
  const estimateLowerSlope = 1 - (medianEstimateError / 100);

  // Labels for specific points
  const labels = [
    {"index": 8, "label": "Missile gap"},
    {"index": 1, "label": "Bomber gap"},
    {"index": 3, "label": "Iraq intelligence failure"}
  ];

  // Create traces for left subplot (Stated ranges)
  const leftTraces = [];
  const uniqueCategories = [...new Set(categories)];
  const websiteColors = [COLORS.purple, COLORS.blue, COLORS.teal, COLORS.dark_teal, COLORS.purple_alt, COLORS.light_teal];
  const categoryColorMap = {};
  uniqueCategories.forEach((cat, i) => {
    categoryColorMap[cat] = websiteColors[i % websiteColors.length];
  });

  // Add error bars and points for each category
  uniqueCategories.forEach(category => {
    const indices = categories.map((c, i) => c === category ? i : -1).filter(i => i !== -1);

    // Add error bar lines
    indices.forEach(i => {
      leftTraces.push({
        x: [centralEstimates[i], centralEstimates[i]],
        y: [lowerBounds[i], upperBounds[i]],
        mode: 'lines',
        line: { color: categoryColorMap[category], width: 1 },
        opacity: 0.3,
        showlegend: false,
        hoverinfo: 'skip',
        xaxis: 'x',
        yaxis: 'y'
      });
    });

    // Add upper bound points
    leftTraces.push({
      x: indices.map(i => centralEstimates[i]),
      y: indices.map(i => upperBounds[i]),
      mode: 'markers',
      marker: { color: categoryColorMap[category], size: 8 },
      opacity: 0.7,
      name: category,
      showlegend: true,
      hovertemplate: 'Central: %{x}<br>Upper: %{y}<extra></extra>',
      xaxis: 'x',
      yaxis: 'y'
    });

    // Add lower bound points
    leftTraces.push({
      x: indices.map(i => centralEstimates[i]),
      y: indices.map(i => lowerBounds[i]),
      mode: 'markers',
      marker: { color: categoryColorMap[category], size: 8 },
      opacity: 0.7,
      showlegend: false,
      hovertemplate: 'Central: %{x}<br>Lower: %{y}<extra></extra>',
      xaxis: 'x',
      yaxis: 'y'
    });
  });

  // Add median error region for left subplot
  const maxRange = Math.max(...centralEstimates, ...upperBounds);
  const minRange = Math.min(...centralEstimates, ...lowerBounds);
  const xLine = linspace(minRange, maxRange, 100);

  leftTraces.push({
    x: [...xLine, ...xLine.reverse()],
    y: [...xLine.map(x => lowerSlope * x), ...xLine.reverse().map(x => upperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianUpperError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Add y=x line for left subplot
  leftTraces.push({
    x: xLine,
    y: xLine,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Create traces for right subplot (Estimate vs reality)
  const rightTraces = [];
  const uniqueEstimateCategories = [...new Set(estimateCategories)];
  const estimateCategoryColorMap = {};
  uniqueEstimateCategories.forEach((cat, i) => {
    estimateCategoryColorMap[cat] = websiteColors[i % websiteColors.length];
  });

  uniqueEstimateCategories.forEach(category => {
    const indices = estimateCategories.map((c, i) => c === category ? i : -1).filter(i => i !== -1);

    rightTraces.push({
      x: indices.map(i => groundTruths[i]),
      y: indices.map(i => estimates[i]),
      mode: 'markers',
      marker: { color: estimateCategoryColorMap[category], size: 8 },
      opacity: 0.7,
      name: category,
      showlegend: true,
      hovertemplate: 'Ground truth: %{x}<br>Estimate: %{y}<extra></extra>',
      xaxis: 'x2',
      yaxis: 'y2'
    });
  });

  // Add median error region for right subplot
  const maxRangeEst = Math.max(...estimates, ...groundTruths);
  const minRangeEst = Math.min(...estimates, ...groundTruths);
  const xLineEst = linspace(minRangeEst, maxRangeEst, 100);

  rightTraces.push({
    x: [...xLineEst, ...xLineEst.reverse()],
    y: [...xLineEst.map(x => estimateLowerSlope * x), ...xLineEst.reverse().map(x => estimateUpperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianEstimateError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip',
    xaxis: 'x2',
    yaxis: 'y2'
  });

  // Add y=x line for right subplot
  rightTraces.push({
    x: xLineEst,
    y: xLineEst,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip',
    xaxis: 'x2',
    yaxis: 'y2'
  });

  // Combine all traces
  const data = [...leftTraces, ...rightTraces];

  // Create layout with subplots
  const layout = {
    height: 300,
    showlegend: true,
    autosize: true,
    grid: {
      rows: 1,
      columns: 2,
      pattern: 'independent',
      xgap: 0.15
    },
    xaxis: {
      title: 'Central Estimate (Midpoint)',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0.18, 0.48]
    },
    yaxis: {
      title: 'Stated estimate range',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray'
    },
    xaxis2: {
      title: 'Ground Truth',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0.63, 0.93]
    },
    yaxis2: {
      title: 'Estimate',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      anchor: 'x2'
    },
    legend: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 1.0,
      xanchor: 'right',
      x: 0.17,
      bgcolor: 'rgba(255, 255, 255, 0.9)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    legend2: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 1.0,
      xanchor: 'right',
      x: 0.62,
      bgcolor: 'rgba(255, 255, 255, 0.9)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    margin: { l: 50, r: 50, t: 40, b: 50 },
    plot_bgcolor: 'white',
    font: { size: 10 },
    annotations: [
      {
        text: 'Stated ranges',
        xref: 'paper',
        yref: 'paper',
        x: 0.33,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      },
      {
        text: 'Estimate vs. ground truth',
        xref: 'paper',
        yref: 'paper',
        x: 0.78,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      }
    ]
  };

  // Add text labels for specific points
  const labelOffsets = {
    8: [10, 20],   // Missile gap
    1: [10, -25],  // Bomber gap
    3: [10, 40]    // Iraq intelligence failure
  };

  labels.forEach(labelInfo => {
    const idx = labelInfo.index;
    const offset = labelOffsets[idx] || [10, 10];

    layout.annotations.push({
      x: groundTruths[idx],
      y: estimates[idx],
      text: labelInfo.label,
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 1,
      arrowcolor: 'gray',
      ax: offset[0],
      ay: offset[1],
      font: { size: 9 },
      bgcolor: 'white',
      bordercolor: 'gray',
      borderwidth: 1,
      borderpad: 3,
      opacity: 0.8,
      xref: 'x2',
      yref: 'y2'
    });
  });

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(elementId, data, layout, config);
}

// Standalone function to create error bars plot
function createErrorBarsPlot(elementId) {
  // Website color scheme - use palette colors
  const COLORS = {
    'purple': COLOR_PALETTE.chip_stock,      // Indigo #6E7FD9
    'blue': COLOR_PALETTE.detection,          // Pewter Blue #8DB8DA
    'teal': COLOR_PALETTE.datacenters_and_energy,  // Viridian #56C4AE
    'dark_teal': '#3D9E8A',                   // Darker viridian
    'red': COLOR_PALETTE.survival_rate,       // Vermillion #F0655A
    'purple_alt': COLOR_PALETTE.chip_stock,   // Indigo #6E7FD9
    'light_teal': '#7DD4C0'                   // Lighter viridian
  };

  // Data for stated error bars (excluding Russian Federation nuclear warheads with min: 1000, max: 2000)
  const statedErrorBars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160},
    {"category": "Nuclear Warheads", "min": 140, "max": 157},
    {"category": "Nuclear Warheads", "min": 225, "max": 300},
    {"category": "Nuclear Warheads", "min": 60, "max": 80},
    {"category": "Fissile material (kg)", "min": 25, "max": 35},
    {"category": "Fissile material (kg)", "min": 30, "max": 50},
    {"category": "Fissile material (kg)", "min": 17, "max": 33},
    {"category": "Fissile material (kg)", "min": 335, "max": 400},
    {"category": "Fissile material (kg)", "min": 330, "max": 580},
    {"category": "Fissile material (kg)", "min": 240, "max": 395},
    {"category": "ICBM launchers", "min": 10, "max": 25},
    {"category": "ICBM launchers", "min": 10, "max": 25},
    {"category": "ICBM launchers", "min": 105, "max": 120},
    {"category": "ICBM launchers", "min": 200, "max": 240},
    {"category": "Intercontinental missiles", "min": 180, "max": 190},
    {"category": "Intercontinental missiles", "min": 200, "max": 300},
    {"category": "Intercontinental missiles", "min": 192, "max": 192}
  ];

  // Calculate central estimates and bounds
  const centralEstimates = [];
  const lowerBounds = [];
  const upperBounds = [];
  const categories = [];
  const upperPercentErrors = [];
  const lowerPercentErrors = [];

  statedErrorBars.forEach(entry => {
    const central = (entry.min + entry.max) / 2;
    centralEstimates.push(central);
    lowerBounds.push(entry.min);
    upperBounds.push(entry.max);
    categories.push(entry.category);

    const upperError = ((entry.max - central) / central) * 100;
    const lowerError = ((central - entry.min) / central) * 100;
    upperPercentErrors.push(upperError);
    lowerPercentErrors.push(lowerError);
  });

  // Calculate median percent errors
  const medianUpperError = median(upperPercentErrors);
  const medianLowerError = median(lowerPercentErrors);

  // Calculate slopes
  const upperSlope = 1 + (medianUpperError / 100);
  const lowerSlope = 1 - (medianLowerError / 100);

  // Data for estimate vs reality
  const estimates = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208];
  const groundTruths = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287];
  const estimateCategories = [
    "Aircraft", "Aircraft", "Aircraft",
    "Chemical Weapons (metric tons)", "Chemical Weapons (metric tons)",
    "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers",
    "Nuclear Warheads (/10)", "Nuclear Warheads (/10)",
    "Ground combat systems (/10)", "Ground combat systems (/10)", "Ground combat systems (/10)",
    "Troops (/1000)"
  ];

  // Calculate median estimate error
  const estimatePercentErrors = [];
  for (let i = 0; i < estimates.length; i++) {
    if (groundTruths[i] !== 0) {
      estimatePercentErrors.push(Math.abs((estimates[i] - groundTruths[i]) / groundTruths[i]) * 100);
    }
  }
  const medianEstimateError = median(estimatePercentErrors);

  const estimateUpperSlope = 1 + (medianEstimateError / 100);
  const estimateLowerSlope = 1 - (medianEstimateError / 100);

  // Labels for specific points
  const labels = [
    {"index": 8, "label": "Missile gap"},
    {"index": 1, "label": "Bomber gap"},
    {"index": 3, "label": "Iraq intelligence failure"}
  ];

  // Create traces for left subplot (Stated ranges)
  const leftTraces = [];
  const uniqueCategories = [...new Set(categories)];
  const websiteColors = [COLORS.purple, COLORS.blue, COLORS.teal, COLORS.dark_teal, COLORS.purple_alt, COLORS.light_teal];
  const categoryColorMap = {};
  uniqueCategories.forEach((cat, i) => {
    categoryColorMap[cat] = websiteColors[i % websiteColors.length];
  });

  // Add error bars and points for each category
  uniqueCategories.forEach(category => {
    const indices = categories.map((c, i) => c === category ? i : -1).filter(i => i !== -1);

    // Add error bar lines
    indices.forEach(i => {
      leftTraces.push({
        x: [centralEstimates[i], centralEstimates[i]],
        y: [lowerBounds[i], upperBounds[i]],
        mode: 'lines',
        line: { color: categoryColorMap[category], width: 1 },
        opacity: 0.3,
        showlegend: false,
        hoverinfo: 'skip',
        xaxis: 'x',
        yaxis: 'y'
      });
    });

    // Add upper bound points
    leftTraces.push({
      x: indices.map(i => centralEstimates[i]),
      y: indices.map(i => upperBounds[i]),
      mode: 'markers',
      marker: { color: categoryColorMap[category], size: 8 },
      opacity: 0.7,
      name: category,
      showlegend: true,
      hovertemplate: 'Central: %{x}<br>Upper: %{y}<extra></extra>',
      xaxis: 'x',
      yaxis: 'y'
    });

    // Add lower bound points
    leftTraces.push({
      x: indices.map(i => centralEstimates[i]),
      y: indices.map(i => lowerBounds[i]),
      mode: 'markers',
      marker: { color: categoryColorMap[category], size: 8 },
      opacity: 0.7,
      showlegend: false,
      hovertemplate: 'Central: %{x}<br>Lower: %{y}<extra></extra>',
      xaxis: 'x',
      yaxis: 'y'
    });
  });

  // Add median error region for left subplot
  const maxRange = Math.max(...centralEstimates, ...upperBounds);
  const minRange = Math.min(...centralEstimates, ...lowerBounds);
  const xLine = linspace(minRange, maxRange, 100);

  leftTraces.push({
    x: [...xLine, ...xLine.reverse()],
    y: [...xLine.map(x => lowerSlope * x), ...xLine.reverse().map(x => upperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianUpperError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Add y=x line for left subplot
  leftTraces.push({
    x: xLine,
    y: xLine,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Create traces for right subplot (Estimate vs reality)
  const rightTraces = [];
  const uniqueEstimateCategories = [...new Set(estimateCategories)];
  const estimateCategoryColorMap = {};
  uniqueEstimateCategories.forEach((cat, i) => {
    estimateCategoryColorMap[cat] = websiteColors[i % websiteColors.length];
  });

  uniqueEstimateCategories.forEach(category => {
    const indices = estimateCategories.map((c, i) => c === category ? i : -1).filter(i => i !== -1);

    rightTraces.push({
      x: indices.map(i => groundTruths[i]),
      y: indices.map(i => estimates[i]),
      mode: 'markers',
      marker: { color: estimateCategoryColorMap[category], size: 8 },
      opacity: 0.7,
      name: category,
      showlegend: true,
      hovertemplate: 'Ground truth: %{x}<br>Estimate: %{y}<extra></extra>',
      xaxis: 'x2',
      yaxis: 'y2'
    });
  });

  // Add median error region for right subplot
  const maxRangeEst = Math.max(...estimates, ...groundTruths);
  const minRangeEst = Math.min(...estimates, ...groundTruths);
  const xLineEst = linspace(minRangeEst, maxRangeEst, 100);

  rightTraces.push({
    x: [...xLineEst, ...xLineEst.reverse()],
    y: [...xLineEst.map(x => estimateLowerSlope * x), ...xLineEst.reverse().map(x => estimateUpperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianEstimateError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip',
    xaxis: 'x2',
    yaxis: 'y2'
  });

  // Add y=x line for right subplot
  rightTraces.push({
    x: xLineEst,
    y: xLineEst,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip',
    xaxis: 'x2',
    yaxis: 'y2'
  });

  // Combine all traces
  const data = [...leftTraces, ...rightTraces];

  // Create layout with subplots
  const layout = {
    height: 300,
    showlegend: true,
    autosize: true,
    grid: {
      rows: 1,
      columns: 2,
      pattern: 'independent',
      xgap: 0.15
    },
    xaxis: {
      title: 'Central Estimate (Midpoint)',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0.18, 0.48]
    },
    yaxis: {
      title: 'Stated estimate range',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray'
    },
    xaxis2: {
      title: 'Ground Truth',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0.63, 0.93]
    },
    yaxis2: {
      title: 'Estimate',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      anchor: 'x2'
    },
    legend: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 1.0,
      xanchor: 'right',
      x: 0.17,
      bgcolor: 'rgba(255, 255, 255, 0.9)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    legend2: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 1.0,
      xanchor: 'right',
      x: 0.62,
      bgcolor: 'rgba(255, 255, 255, 0.9)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    margin: { l: 50, r: 50, t: 40, b: 50 },
    plot_bgcolor: 'white',
    font: { size: 10 },
    annotations: [
      {
        text: 'Stated ranges',
        xref: 'paper',
        yref: 'paper',
        x: 0.205,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      },
      {
        text: 'Estimate vs. ground truth',
        xref: 'paper',
        yref: 'paper',
        x: 0.795,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      }
    ]
  };

  // Add text labels for specific points
  const labelOffsets = {
    8: [10, 20],   // Missile gap
    1: [10, -25],  // Bomber gap
    3: [10, 40]    // Iraq intelligence failure
  };

  labels.forEach(labelInfo => {
    const idx = labelInfo.index;
    const offset = labelOffsets[idx] || [10, 10];

    layout.annotations.push({
      x: groundTruths[idx],
      y: estimates[idx],
      text: labelInfo.label,
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 1,
      arrowcolor: 'gray',
      ax: offset[0],
      ay: offset[1],
      font: { size: 9 },
      bgcolor: 'white',
      bordercolor: 'gray',
      borderwidth: 1,
      borderpad: 3,
      opacity: 0.8,
      xref: 'x2',
      yref: 'y2'
    });
  });

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(elementId, data, layout, config);
}

// Helper functions
function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function linspace(start, end, num) {
  const arr = [];
  const step = (end - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i);
  }
  return arr;
}
