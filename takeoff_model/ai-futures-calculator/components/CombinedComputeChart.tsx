'use client';

import { memo, useMemo, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { ChartDataPoint } from '@/app/types';
import { calculateDynamicXDomain, calculateTicks, createScale, clamp } from '@/utils/chartUtils';
import { formatEffectiveComputeValue, formatYearMonth, formatOOMSuperscriptText, formatPowerOfTenNode } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { COMPUTE_CHART_EXPLANATION } from '@/constants/chartExplanations';
import { WithChartTooltip } from './ChartTitleTooltip';

export interface CombinedComputeChartProps {
  chartData: ChartDataPoint[];
  className?: string;
  title?: string;
  height?: number;
  domainOverride?: [number, number];
  horizontalReferenceLines?: Array<{ y: number; stroke: string; strokeDasharray?: string; strokeWidth?: number; label?: string }>;
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  displayEndYear: number;
  showTrainingSeries?: boolean;
}

export const CombinedComputeChart = memo<CombinedComputeChartProps>(({ chartData, className, title = 'Effective & Training Compute', height, domainOverride, horizontalReferenceLines, verticalReferenceLine, displayEndYear, showTrainingSeries = true }) => {

  const yDomain = useMemo((): [number, number] => {
    const recentData: ChartDataPoint[] = chartData.filter(p => p.year <= displayEndYear);
    const observed: number[] = [
        ...recentData.map(p => p.effectiveCompute).filter((c: number | null | undefined): c is number => c != null && Number.isFinite(c)),
    ];
    if (showTrainingSeries) {
      observed.push(
        ...recentData.map(p => p.trainingCompute).filter((c: number | null | undefined): c is number => c != null && Number.isFinite(c)),
      );
    }

    const maxObserved = observed.length > 0 ? Math.max(...observed) : 0;
    const minObserved = observed.length > 0 ? Math.min(...observed) : 0;
    const suggestedMax = maxObserved + 2;
    const suggestedMin = minObserved - 1;
    if (domainOverride) {
      return domainOverride;
    }
    return [suggestedMin, suggestedMax];
  }, [chartData, displayEndYear, domainOverride, showTrainingSeries]);

  const xDomain = useMemo((): [number, number] => {
    const years = chartData
      .filter(d => {
        const hasEC = d.effectiveCompute != null && !isNaN(d.effectiveCompute as number);
        const hasTC = showTrainingSeries && d.trainingCompute != null && !isNaN(d.trainingCompute as number);
        return hasEC || hasTC;
      })
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, 2030];

    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.05,
      maxPadding: 0.15,
      minRange: 3,
      maxDomain: displayEndYear
    });
  }, [chartData, verticalReferenceLine, displayEndYear, showTrainingSeries]);

  const mergedData = useMemo(() => {
    return chartData.map((point) => ({ ...point, x: point.year }));
  }, [chartData]);

  const lines = useMemo(() => {
    const result: Array<{ dataKey: string; stroke: string; strokeWidth?: number; strokeOpacity?: number; name?: string }> = [];

    result.push({ dataKey: 'effectiveCompute', stroke: '#2A623D', strokeWidth: 3, strokeOpacity: 1, name: 'Effective' });
    if (showTrainingSeries) {
      result.push({ dataKey: 'trainingCompute', stroke: '#888', strokeWidth: 3, strokeOpacity: 1, name: 'Training' });
    }

    return result;
  }, [showTrainingSeries]);

  const yTicks = useMemo(() => {
    return calculateTicks(yDomain, { targetCount: 6 }, 'linear').filter(t => t !== 0);
  }, [yDomain]);

  const tooltip = (point: DataPoint): ReactNode => {
    const year = point['year'] as number | undefined;
    const ec = typeof point['effectiveCompute'] === 'number' ? (point['effectiveCompute'] as number) : null;
    const tc = typeof point['trainingCompute'] === 'number' ? (point['trainingCompute'] as number) : null;

    return (
      <div style={tooltipBoxStyle}>
        {typeof year === 'number' && (
          <span style={tooltipHeaderStyle}>{formatYearMonth(year)}</span>
        )}
        {ec != null && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#2A623D' }} />
            <span style={{ ...tooltipValueStyle, color: '#2A623D' }}>{formatEffectiveComputeValue(ec)} FLOP</span>
          </span>
        )}
        {showTrainingSeries && tc != null && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#888' }} />
            <span style={{ ...tooltipValueStyle, color: '#888' }}>{formatEffectiveComputeValue(tc)} FLOP</span>
          </span>
        )}
      </div>
    );
  };

  // Helper utilities for label positioning and values
  type SeriesKey = 'effectiveCompute' | 'trainingCompute';

  const getSeriesPoints = (key: SeriesKey) => {
    return mergedData
      .filter(d => d[key] != null && !isNaN(d[key] as number))
      .map(d => ({ x: d.x as number, y: d[key] as number }))
      .sort((a, b) => a.x - b.x);
  };

  

  const getVisibleEndpoints = (key: SeriesKey, xMin: number, xMax: number) => {
    const pts = getSeriesPoints(key);
    if (pts.length === 0) return { left: null as null | { x: number; y: number }, right: null as null | { x: number; y: number } };

    const firstX = pts[0].x;
    const lastX = pts[pts.length - 1].x;
    // If series does not overlap visible domain at all
    if (lastX < xMin || firstX > xMax) return { left: null, right: null };

    const leftX = Math.max(xMin, Math.min(xMax, firstX));
    const rightX = Math.min(xMax, Math.max(xMin, lastX));
    const leftY = interpolateAtX(pts, leftX);
    const rightY = interpolateAtX(pts, rightX);

    return {
      left: leftY == null ? null : { x: leftX, y: leftY },
      right: rightY == null ? null : { x: rightX, y: rightY },
    };
  };

  const interpolateAtX = (points: Array<{ x: number; y: number }>, x: number): number | null => {
    if (!points || points.length === 0) return null;
    // If x out of bounds
    if (x <= points[0].x) return points[0].y;
    if (x >= points[points.length - 1].x) return points[points.length - 1].y;

    // Find segment containing x
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      if (x >= p1.x && x <= p2.x && p2.x !== p1.x) {
        const t = (x - p1.x) / (p2.x - p1.x);
        return p1.y + t * (p2.y - p1.y);
      }
    }
    return null;
  };

  const findIntersection = (): { x: number; y: number } | null => {
    if (!showTrainingSeries) return null;
    const ecPts = getSeriesPoints('effectiveCompute');
    const tcPts = getSeriesPoints('trainingCompute');
    if (ecPts.length < 2 || tcPts.length < 2) return null;

    // Build unified sorted x grid across both series
    const xs = Array.from(new Set([...ecPts.map(p => p.x), ...tcPts.map(p => p.x)])).sort((a, b) => a - b);
    if (xs.length < 2) return null;

    // Walk segments for sign change of (ec - tc)
    let prevX: number | null = null;
    let prevDiff: number | null = null;
    for (const x of xs) {
      const ec = interpolateAtX(ecPts, x);
      const tc = interpolateAtX(tcPts, x);
      if (ec == null || tc == null) continue;
      const diff = ec - tc;
      if (prevX != null && prevDiff != null && diff !== prevDiff && (diff === 0 || prevDiff === 0 || diff * prevDiff < 0)) {
        // Linear interpolate zero crossing between prevX and x
        const x1 = prevX;
        const x2 = x;
        const d1 = prevDiff;
        const d2 = diff;
        const denom = (d2 - d1);
        if (denom === 0) continue;
        const t = (0 - d1) / denom;
        const xi = x1 + t * (x2 - x1);
        const eci = interpolateAtX(ecPts, xi);
        const tci = interpolateAtX(tcPts, xi);
        if (eci != null && tci != null) {
          return { x: xi, y: (eci + tci) / 2 };
        }
      }
      prevX = x;
      prevDiff = diff;
    }

    // Fallback to vertical reference line if provided inside domain
    const fallbackX = verticalReferenceLine?.x;
    if (fallbackX != null && isFinite(fallbackX)) {
      const eci = interpolateAtX(ecPts, fallbackX);
      const tci = interpolateAtX(tcPts, fallbackX);
      if (eci != null && tci != null) {
        return { x: fallbackX, y: (eci + tci) / 2 };
      }
    }
    return null;
  };

  return (
    <div className={className || 'flex-1'} style={{ position: 'relative' }}>
      <div className="flex gap-2 items-center">
        <span className="text-[12px] font-semibold text-primary text-left font-system-mono">{title}</span>
        <div className="flex-1 border-t border-gray-500/30" />
        <WithChartTooltip explanation={COMPUTE_CHART_EXPLANATION} className="!gap-0"><></></WithChartTooltip>
      </div>
      <CustomLineChart
        data={mergedData}
        height={height ?? 239}
        width={CHART_LAYOUT.primary.width}
        margin={{ top: 0, right: 60, left: 50, bottom: 35 }}
        xDomain={xDomain}
        xTickCount={5}
        xTickFormatter={(value) => Math.round(value).toString()}
        yDomain={yDomain}
        showYAxis={false}
        yTicks={yTicks}
        yTickFormatter={(v) => formatOOMSuperscriptText(v)}
        showReferencePoints={false}
        lines={lines}
        referenceLines={horizontalReferenceLines}
        tooltip={tooltip}
        customElements={(context) => {
          const { chartWidth, chartHeight } = context;
          const { xDomain: renderedXDomain } = context as typeof context & { xDomain: [number, number] };
          // Create pixel scales
          const xScalePx = createScale({ domain: renderedXDomain, range: [0, chartWidth], type: 'linear' });
          const yScalePx = createScale({ domain: yDomain, range: [chartHeight, 0], type: 'linear' });

          const ecEndpoints = getVisibleEndpoints('effectiveCompute', renderedXDomain[0], renderedXDomain[1]);
          const tcEndpoints = showTrainingSeries
            ? getVisibleEndpoints('trainingCompute', renderedXDomain[0], renderedXDomain[1])
            : { left: null, right: null };
          const intersection = findIntersection();

          const baseFontSize = 12;
          const superscriptFontSize = 12; // keep superscripts size
          const tensFontSize = 12; // slightly smaller tens

          // Compute label positions and avoid overlap with basic nudge
          const elements: ReactNode[] = [];

          

          // Left labels: two-line label (series name on first line, value on second)
          const labelValueOptions = {
            renderMode: 'svg' as const,
            mantissaOpacity: 0.5,
            mantissaFontSize: tensFontSize,
            exponentFontSize: superscriptFontSize,
            suffix: ' FLOP',
            baselineShift: '0.3em' as const,
          };

          const leftLabels: Array<{ pt: { x: number; y: number }; color: string; label: string; value: ReactNode }> = [];
          if (ecEndpoints.left) {
            leftLabels.push({
              pt: ecEndpoints.left,
              color: '#2A623D',
              label: 'Effective',
              value: formatPowerOfTenNode(ecEndpoints.left.y, labelValueOptions)
            });
          }
          if (showTrainingSeries && tcEndpoints.left) {
            leftLabels.push({
              pt: tcEndpoints.left,
              color: '#666',
              label: 'Training',
              value: formatPowerOfTenNode(tcEndpoints.left.y, labelValueOptions)
            });
          }

          let lastLeftY: number | null = null;
          leftLabels.forEach((lbl, idx) => {
            const x = xScalePx(lbl.pt.x);
            let y = yScalePx(lbl.pt.y);
            // Clamp within chart area
            y = clamp(y, 0, chartHeight);
            // Basic anti-overlap
            if (lastLeftY != null && Math.abs(y - lastLeftY) < 16) y += 16 * (idx + 1);
            lastLeftY = y;
            const nearLeft = x < 12;
            const tx = nearLeft ? x + 6 : x - 6;
            const anchor = nearLeft ? 'start' : 'end';
            elements.push(
              <circle key={`left-dot-${idx}`} cx={x} cy={y} r={3} fill={lbl.color} />,
              <text key={`left-${idx}`} x={tx} y={y} textAnchor={anchor} alignmentBaseline="middle" fontSize={baseFontSize} fill={lbl.color}>
                <tspan x={tx} dy={-4}>{lbl.label}</tspan>
                <tspan x={tx} dy={16}>{lbl.value}</tspan>
              </text>
            );
          });

          // Right labels: value only
          const rightLabels: Array<{ pt: { x: number; y: number }; color: string; text: ReactNode }> = [];
          if (ecEndpoints.right) {
            rightLabels.push({
              pt: ecEndpoints.right,
              color: '#2A623D',
              text: formatPowerOfTenNode(ecEndpoints.right.y, labelValueOptions)
            });
          }
          if (showTrainingSeries && tcEndpoints.right) {
            rightLabels.push({
              pt: tcEndpoints.right,
              color: '#666',
              text: formatPowerOfTenNode(tcEndpoints.right.y, labelValueOptions)
            });
          }

          let lastRightY: number | null = null;
          rightLabels.forEach((lbl, idx) => {
            const x = xScalePx(lbl.pt.x);
            let y = yScalePx(lbl.pt.y);
            y = clamp(y, 0, chartHeight);
            if (lastRightY != null && Math.abs(y - lastRightY) < 16) y += 16 * (idx + 1);
            lastRightY = y;
            const nearRight = x > chartWidth - 12;
            const tx = nearRight ? x - 6 : x + 6;
            const anchor = nearRight ? 'end' : 'start';
            elements.push(
              <circle key={`right-dot-${idx}`} cx={x} cy={y} r={3} fill={lbl.color} />,
              <text key={`right-${idx}`} x={tx} y={y} textAnchor={anchor} alignmentBaseline="middle" fontSize={baseFontSize} fill={lbl.color}>
                {lbl.text}
              </text>
            );
          });

          // Crossover label: value only + small marker
          if (showTrainingSeries && intersection) {
            const xi = xScalePx(intersection.x);
            let yi = yScalePx(intersection.y);
            yi = clamp(yi, 0, chartHeight);
            elements.push(
              <g key="crossover">
                <circle cx={xi} cy={yi} r={3} fill="#555" />
                {/* Place label above-left of the point to avoid line overlap */}
                <text x={xi - 6} y={yi - 10} textAnchor="end" fontSize={baseFontSize} fill="#555">{formatPowerOfTenNode(intersection.y, {
                  ...labelValueOptions,
                  mantissaFontSize: tensFontSize,
                  exponentFontSize: superscriptFontSize,
                  suffix: ' FLOP'
                })}</text>
              </g>
            );
          }

          return <g>{elements}</g>;
        }}
      />
    </div>
  );
});

CombinedComputeChart.displayName = 'CombinedComputeChart';

export default CombinedComputeChart;


