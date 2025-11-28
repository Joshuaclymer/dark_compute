'use client';
import { ChartDataPoint } from '@/app/types';
import { memo, useMemo, useEffect, useRef, useState, useCallback, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { formatTo3SigFigs, formatAsPowerOfTenText, formatPowerOfTenNode } from '@/utils/formatting';
import { calculateDynamicXDomain, clamp, createScale, ChartMargin } from '@/utils/chartUtils';
import { CHART_LAYOUT } from '@/constants/chartLayout';

export interface CustomMetricChartProps {
  chartData: ChartDataPoint[];
  tooltip: (point: DataPoint) => ReactNode;
  className?: string;
  title: string;
  dataKey: string;
  yFormatter?: (value: number) => string | ReactNode;
  yScale?: 'linear' | 'log';
  color?: string;
  domainOverride?: [number, number] | ((observed: number[]) => [number, number]);
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  displayEndYear: number;
  height?: number;
  logSuffix?: string;
  width?: number;
  showXAxis?: boolean;
  xDomainOverride?: [number, number];
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  xLabel?: string;
  margin?: ChartMargin;
  showTitle?: boolean;
}

function CustomMetricChartComponent({
  chartData,
  tooltip,
  className,
  title,
  dataKey,
  yFormatter,
  yScale = 'linear',
  color = '#2A623D',
  domainOverride,
  verticalReferenceLine,
  displayEndYear,
  height,
  logSuffix,
  width,
  showXAxis,
  xDomainOverride,
  xTickCount,
  xTickFormatter,
  xLabel,
  margin,
  showTitle = true,
}: CustomMetricChartProps) {
  const resolvedYFormatter = useMemo<(value: number) => string | ReactNode>(() => {
    if (yFormatter) {
      return yFormatter;
    }

    return yScale === 'log'
      ? (value: number) => formatAsPowerOfTenText(value)
      : (value: number) => formatTo3SigFigs(value);
  }, [yFormatter, yScale]);

  const containerRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const [chartHeight, setChartHeight] = useState<number>(() => height ?? 0);

  useEffect(() => {
    if (height != null) {
      setChartHeight(height);
      return;
    }

    const updateHeight = () => {
      if (!containerRef.current) return;

      const containerHeight = containerRef.current.clientHeight;
      setChartHeight(containerHeight);
    };

    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [height]);

  const yDomain = useMemo(() => {
    const observed: number[] = [];

    // Collect all values for this metric
    chartData.forEach(point => {
      const value = point[dataKey as keyof ChartDataPoint];
      if (typeof value === 'number' && Number.isFinite(value) && point.year <= displayEndYear) {
        observed.push(value);
      }
    });

    if (observed.length === 0) {
      return [0, 1] as [number, number];
    }

    const minObserved = Math.min(...observed);
    const maxObserved = Math.max(...observed);

    let domain: [number, number];

    if (domainOverride) {
      domain = typeof domainOverride === 'function' ? domainOverride(observed) : domainOverride;
    } else if (yScale === 'log') {
      const positiveValues = observed.filter(v => v > 0);
      if (positiveValues.length === 0) {
        domain = [1, 10];
      } else {
        const minPositive = Math.min(...positiveValues);
        const maxPositive = Math.max(...positiveValues);
        let lower = minPositive * 0.9;
        const upper = maxPositive * 1.1;

        if (lower <= 0) {
          lower = minPositive * 0.1;
        }

        const safeLower = Math.max(lower, Number.MIN_VALUE);
        let safeUpper = Math.max(upper, safeLower * 1.1);

        if (!Number.isFinite(safeUpper) || safeUpper <= safeLower) {
          safeUpper = safeLower * 10;
        }

        domain = [safeLower, safeUpper];
      }
    } else {
      const suggestedMax = maxObserved > 1 ? Math.floor(Math.max(1, Math.ceil(maxObserved * 1.1))) : 1;
      domain = [Math.min(0, minObserved), suggestedMax];
    }

    return domain;
  }, [chartData, dataKey, yScale, domainOverride, displayEndYear]);

  const xDomain = useMemo((): [number, number] => {
    if (xDomainOverride && xDomainOverride[1] > xDomainOverride[0]) {
      return xDomainOverride;
    }

    const years = chartData
      .filter(d => {
        const value = d[dataKey as keyof ChartDataPoint];
        return value != null && !isNaN(value as number);
      })
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, 2030];

    // Use dynamic domain calculation that considers vertical reference line (intersection)
    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.05,
      maxPadding: 0.15,
      minRange: 3, // Minimum 3 years visible
      maxDomain: displayEndYear // Constrain to smart display end year
    });
  }, [chartData, dataKey, verticalReferenceLine, displayEndYear, xDomainOverride]);

  // Prepare trajectory lines
  const trajectoryKeys = useMemo(() => {
    if (chartData.length === 0) return [];
    const sample = chartData[0];
    return Object.keys(sample).filter(key => key.startsWith(`${dataKey}_trajectory_`));
  }, [chartData, dataKey]);

  // Add x property to chartData
  const mergedData = useMemo<(ChartDataPoint & { x: number })[]>(() => {
    const withinDisplay = chartData.filter(point => point.year <= displayEndYear);
    const source = withinDisplay.length > 0 ? withinDisplay : chartData;
    return source.map(point => ({ ...point, x: point.year }));
  }, [chartData, displayEndYear]);

  // Build line configs
  const lines = useMemo(() => {
    const result = [];

    // Add trajectory lines first (so they're behind)
    for (const key of trajectoryKeys) {
      result.push({
        dataKey: key,
        stroke: color,
        strokeWidth: 1,
        strokeOpacity: 0.3,
      });
    }

    // Add main line
    result.push({
      dataKey: dataKey,
      stroke: color,
      strokeWidth: 3,
      strokeOpacity: 1,
      name: title,
    });

    return result;
  }, [trajectoryKeys, dataKey, color, title]);

  type XYPoint = { x: number; y: number };
type EndpointLabelRendererContext = {
  chartWidth: number;
  chartHeight: number;
  xScale: (value: number) => number;
  yScale: (value: number) => number;
  xDomain?: [number, number];
  yDomain?: [number, number];
};

type EndpointLabelRenderer = ((context: EndpointLabelRendererContext) => ReactNode) & { displayName?: string };

type PixelPoint = { x: number; y: number };
type PixelSegment = { a: PixelPoint; b: PixelPoint };

const LABEL_HEIGHT = 16;
const DEFAULT_LABEL_WIDTH = 60;
const LABEL_HORIZONTAL_GAP = 25;
const LABEL_VERTICAL_GAP = 16;
const LABEL_PADDING = 4;
const SEGMENTS_TO_CHECK = 20;
const SEGMENT_SAMPLE_STEPS = 6;
const OVERLAP_PADDING = 2;

const estimateLabelWidth = (label: ReactNode): number => {
  if (typeof label === 'number') {
    return clamp(String(label).length * 6 + 12, 36, 140);
  }
  if (typeof label === 'string') {
    return clamp(label.length * 6 + 14, 36, 160);
  }
  if (Array.isArray(label)) {
    const joined = label
      .map((child) => (typeof child === 'string' || typeof child === 'number' ? String(child) : ''))
      .join('');
    if (joined) {
      return clamp(joined.length * 6 + 14, 36, 160);
    }
  }
  return DEFAULT_LABEL_WIDTH;
};

const computeLabelBounds = (
  textX: number,
  textY: number,
  anchor: 'start' | 'middle' | 'end',
  width: number,
  height: number
) => {
  let x1: number;
  let x2: number;
  if (anchor === 'start') {
    x1 = textX;
    x2 = textX + width;
  } else if (anchor === 'end') {
    x1 = textX - width;
    x2 = textX;
  } else {
    x1 = textX - width / 2;
    x2 = textX + width / 2;
  }

  const y1 = textY - height / 2;
  const y2 = textY + height / 2;
  return { x1, x2, y1, y2 };
};

const clampLabelWithinChart = (
  candidate: { textX: number; textY: number; anchor: 'start' | 'middle' | 'end' },
  width: number,
  height: number,
  chartWidth: number,
  chartHeight: number
) => {
  const { anchor } = candidate;
  let { textX, textY } = candidate;

  let bounds = computeLabelBounds(textX, textY, anchor, width, height);

  if (bounds.x1 < LABEL_PADDING) {
    const delta = LABEL_PADDING - bounds.x1;
    textX += delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  } else if (bounds.x2 > chartWidth - LABEL_PADDING) {
    const delta = bounds.x2 - (chartWidth - LABEL_PADDING);
    textX -= delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  }

  if (bounds.y1 < LABEL_PADDING) {
    const delta = LABEL_PADDING - bounds.y1;
    textY += delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  } else if (bounds.y2 > chartHeight - LABEL_PADDING) {
    const delta = bounds.y2 - (chartHeight - LABEL_PADDING);
    textY -= delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  }

  return { textX, textY, anchor };
};

const pointInsideRect = (x: number, y: number, rect: { x1: number; x2: number; y1: number; y2: number }) =>
  x >= rect.x1 && x <= rect.x2 && y >= rect.y1 && y <= rect.y2;

const segmentIntersectsRect = (segment: PixelSegment, rect: { x1: number; x2: number; y1: number; y2: number }) => {
  if (
    Math.max(segment.a.x, segment.b.x) < rect.x1 - OVERLAP_PADDING ||
    Math.min(segment.a.x, segment.b.x) > rect.x2 + OVERLAP_PADDING ||
    Math.max(segment.a.y, segment.b.y) < rect.y1 - OVERLAP_PADDING ||
    Math.min(segment.a.y, segment.b.y) > rect.y2 + OVERLAP_PADDING
  ) {
    return false;
  }

  for (let i = 0; i <= SEGMENT_SAMPLE_STEPS; i++) {
    const t = i / SEGMENT_SAMPLE_STEPS;
    const sampleX = segment.a.x + (segment.b.x - segment.a.x) * t;
    const sampleY = segment.a.y + (segment.b.y - segment.a.y) * t;
    if (pointInsideRect(sampleX, sampleY, rect)) {
      return true;
    }
  }

  return false;
};

const doesCandidateOverlap = (
  candidate: { textX: number; textY: number; anchor: 'start' | 'middle' | 'end' },
  width: number,
  height: number,
  segments: PixelSegment[]
) => {
  if (!segments.length) {
    return false;
  }
  const bounds = computeLabelBounds(candidate.textX, candidate.textY, candidate.anchor, width, height);
  const expandedBounds = {
    x1: bounds.x1 - OVERLAP_PADDING,
    x2: bounds.x2 + OVERLAP_PADDING,
    y1: bounds.y1 - OVERLAP_PADDING,
    y2: bounds.y2 + OVERLAP_PADDING,
  };

  return segments.some(segment => segmentIntersectsRect(segment, expandedBounds));
};

  const interpolateAtX = useCallback((points: XYPoint[], x: number): number | null => {
    if (!points || points.length === 0) return null;
    if (x <= points[0].x) return points[0].y;
    if (x >= points[points.length - 1].x) return points[points.length - 1].y;

    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      if (x >= p1.x && x <= p2.x && p2.x !== p1.x) {
        const t = (x - p1.x) / (p2.x - p1.x);
        return p1.y + t * (p2.y - p1.y);
      }
    }

    return null;
  }, []);

  const getVisibleEndpoints = useCallback((points: XYPoint[], xMin: number, xMax: number) => {
    if (!points || points.length === 0) {
      return { left: null as XYPoint | null, right: null as XYPoint | null };
    }

    const firstX = points[0].x;
    const lastX = points[points.length - 1].x;

    if (lastX < xMin || firstX > xMax) {
      return { left: null, right: null };
    }

    const leftX = Math.max(xMin, Math.min(xMax, firstX));
    const rightX = Math.min(xMax, Math.max(xMin, lastX));

    const leftY = interpolateAtX(points, leftX);
    const rightY = interpolateAtX(points, rightX);

    return {
      left: leftY == null ? null : { x: leftX, y: leftY },
      right: rightY == null ? null : { x: rightX, y: rightY },
    };
  }, [interpolateAtX]);

  const formatLogEndpoint = useCallback((value: number): ReactNode => {
    if (!Number.isFinite(value) || value <= 0) {
      return '';
    }

    const exponent = Math.round(Math.log10(value));

    return formatPowerOfTenNode(exponent, {
      renderMode: 'svg',
      mantissaOpacity: 0.7,
      mantissaFontSize: 12,
      exponentFontSize: 12,
      suffix: logSuffix,
    });
  }, [logSuffix]);

  const formatEndpointValue = useCallback((value: number): ReactNode => {
    if (!Number.isFinite(value)) {
      return '';
    }

    if (yScale === 'log') {
      return formatLogEndpoint(value);
    }

    if (yFormatter) {
      const formatted = yFormatter(value);
      if (typeof formatted === 'string' || typeof formatted === 'number') {
        return formatted;
      }
      return formatted;
    }

    return formatTo3SigFigs(value);
  }, [yFormatter, yScale, formatLogEndpoint]);

  const customElements = useMemo(() => {
    const seriesPoints: XYPoint[] = mergedData
      .filter(point => {
        const value = point[dataKey as keyof ChartDataPoint];
        return typeof value === 'number' && Number.isFinite(value) && point.x <= displayEndYear;
      })
      .map(point => ({ x: point.x as number, y: point[dataKey as keyof ChartDataPoint] as number }))
      .sort((a, b) => a.x - b.x);

    if (seriesPoints.length === 0) {
      return undefined;
    }

    const EndpointLabels: EndpointLabelRenderer = (context) => {
      const { chartWidth, chartHeight, xScale, yScale } = context;
      const renderedXDomain =
        context.xDomain ?? ([seriesPoints[0]?.x ?? 0, seriesPoints[seriesPoints.length - 1]?.x ?? 1] as [number, number]);

      const { left, right } = getVisibleEndpoints(seriesPoints, renderedXDomain[0], renderedXDomain[1]);
      const elements: ReactNode[] = [];

      let visibleSeries = seriesPoints.filter(
        point => point.x >= renderedXDomain[0] - 1e-6 && point.x <= renderedXDomain[1] + 1e-6
      );
      if (visibleSeries.length === 0) {
        visibleSeries = [...seriesPoints];
      }
      if (left && (visibleSeries.length === 0 || Math.abs(visibleSeries[0].x - left.x) > 1e-6)) {
        visibleSeries = [left, ...visibleSeries];
      }
      if (right && (visibleSeries.length === 0 || Math.abs(visibleSeries[visibleSeries.length - 1].x - right.x) > 1e-6)) {
        visibleSeries = [...visibleSeries, right];
      }

      const pixelPoints: PixelPoint[] = visibleSeries
        .map(point => ({
          x: xScale(point.x),
          y: clamp(yScale(point.y), -LABEL_HEIGHT, chartHeight + LABEL_HEIGHT),
        }))
        .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));

      const segments: PixelSegment[] = [];
      for (let i = 0; i < pixelPoints.length - 1; i++) {
        segments.push({ a: pixelPoints[i], b: pixelPoints[i + 1] });
      }

      const leftSegments = segments.slice(0, SEGMENTS_TO_CHECK);
      const rightSegments = segments.slice(Math.max(segments.length - SEGMENTS_TO_CHECK, 0));

      const createCandidates = (
        align: 'left' | 'right',
        baseX: number,
        baseY: number,
        labelWidth: number,
        labelHeight: number
      ) => {
        const candidates: Array<{ textX: number; textY: number; anchor: 'start' | 'middle' | 'end' }> = [];
        const add = (anchor: 'start' | 'middle' | 'end', offsetX: number, offsetY: number) => {
          const textX = baseX + offsetX;
          const textY = clamp(baseY + offsetY, labelHeight / 2 + LABEL_PADDING, chartHeight - labelHeight / 2 - LABEL_PADDING);
          candidates.push(clampLabelWithinChart({ textX, textY, anchor }, labelWidth, labelHeight, chartWidth, chartHeight));
        };

        if (align === 'left') {
          add('middle', 0, -LABEL_VERTICAL_GAP);
          add('middle', 0, LABEL_VERTICAL_GAP);
          add('start', LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP / 2);
          add('start', LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP / 2);
          add('end', -LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP / 2);
          add('end', -LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP / 2);
        } else {
          add('start', LABEL_HORIZONTAL_GAP, 0);
          add('start', LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP);
          add('start', LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP);
          add('end', -LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP);
          add('end', -LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP);
          add('middle', 0, -LABEL_VERTICAL_GAP);
          add('middle', 0, LABEL_VERTICAL_GAP);
        }

        return candidates;
      };

      const selectLabelPosition = (
        align: 'left' | 'right',
        baseX: number,
        baseY: number,
        labelWidth: number,
        labelHeight: number,
        segmentsToTest: PixelSegment[]
      ) => {
        const candidates = createCandidates(align, baseX, baseY, labelWidth, labelHeight);
        for (const candidate of candidates) {
          if (!doesCandidateOverlap(candidate, labelWidth, labelHeight, segmentsToTest)) {
            return candidate;
          }
        }
        return candidates[0];
      };

      const renderEndpoint = (endpoint: XYPoint | null, align: 'left' | 'right', key: string) => {
        if (!endpoint) {
          return;
        }

        const xPos = xScale(endpoint.x);
        const yPosRaw = yScale(endpoint.y);

        if (!Number.isFinite(xPos) || !Number.isFinite(yPosRaw)) {
          return;
        }

        const yPos = clamp(yPosRaw, 0, chartHeight);

        const label = formatEndpointValue(endpoint.y);
        if (label == null || label === '') {
          return;
        }

        const labelWidth = estimateLabelWidth(label);
        const placement = selectLabelPosition(
          align,
          xPos,
          yPos,
          labelWidth,
          LABEL_HEIGHT,
          align === 'left' ? leftSegments : rightSegments
        );

        console.log({chartTitle: title, key, placement});

        elements.push(
          <g key={key}>
            <circle cx={xPos} cy={yPos} r={3} fill={color} />
            <text
              x={placement.textX}
              y={placement.textY}
              textAnchor={placement.anchor}
              alignmentBaseline="middle"
              fontSize={11}
              fill={color}
              fontWeight="500"
            >
              {label}
            </text>
          </g>
        );
      };

      renderEndpoint(left, 'left', 'metric-left-endpoint');
      renderEndpoint(right, 'right', 'metric-right-endpoint');

      return <g>{elements}</g>;
    };

    EndpointLabels.displayName = 'CustomMetricChartEndpointLabels';

    return EndpointLabels;
  }, [mergedData, dataKey, displayEndYear, yScale, getVisibleEndpoints, formatEndpointValue, color]);

  return (
    <div
      ref={containerRef}
      className={`flex min-h-0 flex-col relative ${className ? className : ''}`.trim()}
      style={{ height: '100%' }}
    >
      {showTitle ? (
        <h3
          ref={headerRef}
          className="text-[9px] uppercase font-light text-gray-700 font-system-mono absolute left-0 max-w-40"
        >
          {title}
        </h3>
      ) : null}
      <div className="flex-1 min-h-0">
        <CustomLineChart
          data={mergedData}
          height={chartHeight}
          margin={margin ?? { top: 3, right: 10, left: 4, bottom: 3 }}
          xDomain={xDomain}
          xTickCount={xTickCount ?? 5}
          xTickFormatter={xTickFormatter ?? ((value) => Math.round(value).toString())}
          yDomain={yDomain}
          yScale={yScale}
          yTickFormatter={resolvedYFormatter}
          showReferencePoints={false}
          lines={lines}
          tooltip={tooltip}
          width={width ?? CHART_LAYOUT.metric.width}
          showXAxis={showXAxis ?? false}
          xLabel={xLabel}
          customElements={customElements}
        />
      </div>
    </div>
  );
}

export const CustomMetricChart = memo(CustomMetricChartComponent);

CustomMetricChart.displayName = 'CustomMetricChart';
