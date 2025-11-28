'use client';

import { useState, useRef, useCallback, ReactNode, useEffect, useMemo } from 'react';
import { createScale, generateLinePath, calculateTicks, ChartMargin, clamp } from '@/utils/chartUtils';
import { useChartSync } from './ChartSyncContext';

const DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS = 450;

export interface DataPoint {
  x: number;
  y?: number | null;
  [key: string]: number | null | undefined | string;
}

export interface CustomLineChartProps {
  data: DataPoint[];
  width?: number;
  height: number;
  margin?: ChartMargin;

  // X axis config
  xDomain: [number, number];
  showXAxis?: boolean;
  xTickStepSize?: number;
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  xLabel?: string;
  xPadding?: {
    start?: number;
    end?: number;
  };

  // Y axis config
  yDomain: [number, number];
  yScale?: 'linear' | 'log';
  yTicks?: number[];
  yTickFormatter?: (value: number, workTimeLabel?: boolean) => string | ReactNode;
  showYAxis?: boolean;

  // Line configs
  lines: Array<{
    dataKey: string;
    stroke: string;
    strokeWidth?: number;
    strokeOpacity?: number;
    name?: string;
    smooth?: boolean;
  }>;

  // Optional features
  showGrid?: boolean;
  referenceLine?: {
    y: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  referenceLines?: Array<{
    y: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  }>;
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  showReferencePoints?: boolean;
  customElements?: (context: { chartWidth: number; chartHeight: number, xScale: (number: number) => number, yScale: (number: number) => number }) => ReactNode;
  tooltip?: (point: DataPoint) => ReactNode;
  animateXDomain?: boolean;
  xDomainAnimationDurationMs?: number;
}

export function CustomLineChart({
  data,
  width,
  height,
  margin = { top: 20, right: 30, left: 0, bottom: 60 },
  xDomain,
  xTickStepSize,
  xTickCount,
  xTickFormatter = (v) => v.toString(),
  xLabel,
  yDomain,
  yScale = 'linear',
  yTicks,
  yTickFormatter,
  showYAxis = false,
  showXAxis = true,
  lines,
  referenceLine,
  referenceLines,
  verticalReferenceLine,
  showReferencePoints = false,
  customElements,
  tooltip,
  xPadding,
  animateXDomain = true,
  xDomainAnimationDurationMs = DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS,
}: CustomLineChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(width || 600);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { hoveredX, setHoveredX } = useChartSync();
  const [animatedXDomain, setAnimatedXDomain] = useState<[number, number]>(xDomain);
  const animatedXDomainRef = useRef(animatedXDomain);
  const xDomainAnimationFrameRef = useRef<number | null>(null);
  const xDomainAnimationStartRef = useRef(0);
  const xDomainFromRef = useRef<[number, number]>(xDomain);
  const xDomainToRef = useRef<[number, number]>(xDomain);
  const [animatedXTicks, setAnimatedXTicks] = useState<number[]>(() =>
    calculateTicks(
      xDomain,
      xTickCount ? { targetCount: xTickCount } : xTickStepSize ? { stepSize: xTickStepSize } : { targetCount: 3 },
      'linear'
    )
  );
  const animatedXTicksRef = useRef(animatedXTicks);
  const xTicksFromRef = useRef<number[]>(animatedXTicks);
  const xTicksToRef = useRef<number[]>(animatedXTicks);

  useEffect(() => {
    animatedXDomainRef.current = animatedXDomain;
  }, [animatedXDomain]);

  useEffect(() => {
    animatedXTicksRef.current = animatedXTicks;
  }, [animatedXTicks]);

  const xTickOptions = useMemo(() => {
    if (xTickCount) {
      return { targetCount: xTickCount };
    }
    if (xTickStepSize) {
      return { stepSize: xTickStepSize };
    }
    return { targetCount: 3 };
  }, [xTickCount, xTickStepSize]);

  const generateTicksForDomain = useCallback(
    (domain: [number, number]) => calculateTicks(domain, xTickOptions, 'linear'),
    [xTickOptions]
  );

  const sampleTickArray = useCallback((ticks: number[], normalizedIndex: number) => {
    if (!ticks.length) {
      return NaN;
    }
    if (ticks.length === 1) {
      return ticks[0];
    }
    const clampedNorm = clamp(normalizedIndex, 0, 1);
    const scaledIndex = clampedNorm * (ticks.length - 1);
    const lowerIndex = Math.floor(scaledIndex);
    const upperIndex = Math.min(ticks.length - 1, Math.ceil(scaledIndex));
    if (lowerIndex === upperIndex) {
      return ticks[lowerIndex];
    }
    const fraction = scaledIndex - lowerIndex;
    const lowerValue = ticks[lowerIndex];
    const upperValue = ticks[upperIndex];
    return lowerValue + (upperValue - lowerValue) * fraction;
  }, []);

  const interpolateTickArrays = useCallback(
    (from: number[], to: number[], progress: number) => {
      if (!from.length && !to.length) {
        return [];
      }

      if (progress >= 1) {
        return [...to];
      }

      if (progress <= 0) {
        return [...from];
      }

      const maxLength = Math.max(from.length, to.length, 1);
      const result: number[] = [];

      for (let i = 0; i < maxLength; i += 1) {
        const normalizedIndex = maxLength === 1 ? 0 : i / (maxLength - 1);
        const startValueRaw = sampleTickArray(from, normalizedIndex);
        const endValueRaw = sampleTickArray(to, normalizedIndex);
        const fallback = Number.isNaN(startValueRaw) ? endValueRaw : startValueRaw;
        const startValue = Number.isNaN(startValueRaw) ? fallback : startValueRaw;
        const endValue = Number.isNaN(endValueRaw) ? fallback : endValueRaw;
        const safeStart = Number.isFinite(startValue) ? startValue : 0;
        const safeEnd = Number.isFinite(endValue) ? endValue : safeStart;
        result.push(safeStart + (safeEnd - safeStart) * progress);
      }

      return result;
    },
    [sampleTickArray]
  );

  useEffect(() => {
    const arraysEqual = (a: number[], b: number[]) =>
      a.length === b.length && a.every((value, idx) => value === b[idx]);

    const nextTicks = generateTicksForDomain(xDomain);

    if (!animateXDomain) {
      if (
        animatedXDomainRef.current[0] !== xDomain[0] ||
        animatedXDomainRef.current[1] !== xDomain[1]
      ) {
        setAnimatedXDomain(xDomain);
        animatedXDomainRef.current = xDomain;
      }
      xDomainFromRef.current = xDomain;
      xDomainToRef.current = xDomain;
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    const fromDomain = animatedXDomainRef.current;
    const domainUnchanged =
      fromDomain[0] === xDomain[0] && fromDomain[1] === xDomain[1];

    if (domainUnchanged) {
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    xDomainFromRef.current = fromDomain;
    xDomainToRef.current = xDomain;
    xDomainAnimationStartRef.current = performance.now();
    xTicksFromRef.current = animatedXTicksRef.current;
    xTicksToRef.current = nextTicks;

    if (xDomainAnimationFrameRef.current !== null) {
      cancelAnimationFrame(xDomainAnimationFrameRef.current);
    }

    const duration = Math.max(0, xDomainAnimationDurationMs);

    const step = (timestamp: number) => {
      const elapsed = timestamp - xDomainAnimationStartRef.current;
      const progress = duration === 0 ? 1 : Math.min(1, elapsed / duration);
      const nextDomain: [number, number] = [
        xDomainFromRef.current[0] + (xDomainToRef.current[0] - xDomainFromRef.current[0]) * progress,
        xDomainFromRef.current[1] + (xDomainToRef.current[1] - xDomainFromRef.current[1]) * progress,
      ];
      setAnimatedXDomain(nextDomain);
      animatedXDomainRef.current = nextDomain;

      const interpolatedTicks = interpolateTickArrays(
        xTicksFromRef.current,
        xTicksToRef.current,
        progress
      );
      setAnimatedXTicks(interpolatedTicks);
      animatedXTicksRef.current = interpolatedTicks;

      if (progress < 1) {
        xDomainAnimationFrameRef.current = requestAnimationFrame(step);
      } else {
        setAnimatedXTicks(xTicksToRef.current);
        animatedXTicksRef.current = xTicksToRef.current;
        xDomainAnimationFrameRef.current = null;
      }
    };

    xDomainAnimationFrameRef.current = requestAnimationFrame(step);

    return () => {
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
    };
  }, [xDomain, animateXDomain, xDomainAnimationDurationMs, generateTicksForDomain, interpolateTickArrays]);

  // Measure container width for responsive sizing
  useEffect(() => {
    if (!containerRef.current || width) return; // Skip if width is explicitly provided

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [width]);

  const actualWidth = width || containerWidth;
  const chartWidth = actualWidth - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  

  const currentXDomain = animateXDomain ? animatedXDomain : xDomain;

  // Create scales
  const paddingStart = clamp(xPadding?.start ?? 0, 0, chartWidth);
  const paddingEnd = clamp(xPadding?.end ?? 0, 0, chartWidth - paddingStart);
  const xRangeStart = paddingStart;
  const xRangeEnd = Math.max(xRangeStart + 1, chartWidth - paddingEnd);

  const xScale = createScale({
    domain: currentXDomain,
    range: [xRangeStart, xRangeEnd],
    type: 'linear',
  });

  const yScaleFunc = createScale({
    domain: yDomain,
    range: [chartHeight, 0], // SVG Y is inverted
    type: yScale,
  });

  const customElementGroup = customElements
    ? customElements(
        {
          chartWidth,
          chartHeight,
          xDomain: currentXDomain,
          yDomain,
          xScale,
          yScale: yScaleFunc,
        } as { chartWidth: number; chartHeight: number, xScale: (number: number) => number  , yScale: (number: number) => number }
      )
    : null;

  

  // Calculate ticks
  const xTicks = animateXDomain ? animatedXTicks : generateTicksForDomain(currentXDomain);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - margin.left;

    setMousePosition({ x: e.clientX, y: e.clientY });

    // Find closest point based on X-axis distance only (better for time-series)
    const mainDataKey = lines[lines.length - 1]?.dataKey; // Use last line (main line)
    if (!mainDataKey) return;

    const dataWithMainLine = data.filter(d => d[mainDataKey] != null);

    // Find closest point by X-axis distance only
    let closest: DataPoint | null = null;
    let minXDistance = Infinity;

    for (const point of dataWithMainLine) {
      const px = xScale(point.x);
      const xDistance = Math.abs(mouseX - px);

      if (xDistance < minXDistance) {
        minXDistance = xDistance;
        closest = point;
      }
    }

    if (tooltip) {
      setHoveredPoint(closest);
    }

    // Update shared hover X position for crosshair sync
    if (closest) {
      setHoveredX(closest.x);
    }
  }, [data, lines, margin, xScale, tooltip, setHoveredX]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    setMousePosition(null);
    setHoveredX(null);
  }, [setHoveredX]);

  return (
    <div ref={containerRef} style={{ position: 'relative', width: width || '100%', height }}>
      <svg
        ref={svgRef}
        width={actualWidth}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ userSelect: 'none' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Horizontal reference line */}
          {/* {referenceLine && (
            <g>
              <line
                x1={0}
                y1={yScaleFunc(referenceLine.y)}
                x2={chartWidth}
                y2={yScaleFunc(referenceLine.y)}
                stroke={referenceLine.stroke}
                strokeDasharray={referenceLine.strokeDasharray || '8 8'}
                strokeWidth={referenceLine.strokeWidth || 2}
                strokeOpacity={referenceLine.strokeOpacity ?? 1}
                strokeLinecap="round"
              />
              {referenceLine.label && (
                <text
                  x={chartWidth / 2}
                  y={yScaleFunc(referenceLine.y) - 5}
                  textAnchor="middle"
                  fontSize={12}
                  fill={referenceLine.stroke}
                >
                  {referenceLine.label}
                </text>
              )}
            </g>
          )} */}

          {/* Multiple horizontal reference lines */}
          {/* {referenceLines && referenceLines.length > 0 && referenceLines.map((rl, i) => (
            <g id={`multi-ref-${i}`} key={`multi-ref-${i}`}>
              <line
                x1={0}
                y1={yScaleFunc(rl.y)}
                x2={chartWidth}
                y2={yScaleFunc(rl.y)}
                stroke={rl.stroke}
                strokeDasharray={rl.strokeDasharray || '8 8'}
                strokeWidth={rl.strokeWidth || 2}
                strokeOpacity={rl.strokeOpacity ?? 1}
                strokeLinecap="round"
              />
              {rl.label && (
                <text
                  x={chartWidth / 2}
                  y={yScaleFunc(rl.y) - 5}
                  textAnchor="middle"
                  fontSize={12}
                  fill={rl.stroke}
                >
                  {rl.label}
                </text>
              )}
            </g>
          ))} */}

          {/* Vertical reference line */}
          {verticalReferenceLine && (
            <g>
              <line
                x1={xScale(verticalReferenceLine.x)}
                y1={0}
                x2={xScale(verticalReferenceLine.x)}
                y2={chartHeight}
                stroke={verticalReferenceLine.stroke}
                strokeDasharray={verticalReferenceLine.strokeDasharray || '2 10'}
                strokeWidth={verticalReferenceLine.strokeWidth || 1}
                strokeOpacity={verticalReferenceLine.strokeOpacity ?? 0.5}
                strokeLinecap="round"
              />
              {verticalReferenceLine.label && (
                <text
                  x={xScale(verticalReferenceLine.x) + 5}
                  y={15}
                  textAnchor="start"
                  fontSize={11}
                  fill={verticalReferenceLine.stroke}
                  transform={`rotate(-90, ${xScale(verticalReferenceLine.x) + 5}, 15)`}
                >
                  {verticalReferenceLine.label}
                </text>
              )}
            </g>
          )}

          {/* Area fills under lines */}
          {/* {lines.map((line, i) => {
            const lineData = data.map(d => ({
              x: d.x,
              y: d[line.dataKey] as number | null | undefined,
            }));

            // Generate area path (filled to bottom)
            const validPoints = lineData.filter(d => d.y != null && !isNaN(d.y));
            if (validPoints.length === 0) return null;

            const areaPath = validPoints.map((point, idx) => {
              const x = xScale(point.x);
              const y = yScaleFunc(point.y!);
              if (idx === 0) {
                return `M ${x} ${chartHeight} L ${x} ${y}`;
              }
              return `L ${x} ${y}`;
            }).join(' ') + ` L ${xScale(validPoints[validPoints.length - 1].x)} ${chartHeight} Z`;

            return (
              <path
                id={`area-${i}-${line.dataKey}`}
                key={`area-${i}-${line.dataKey}`}
                d={areaPath}
                fill={line.stroke}
                fillOpacity={0.1}
                stroke="none"
              />
            );
          })} */}

          {/* Lines */}
          {lines.map((line, i) => {
            const lineData = data.map(d => ({
              x: d.x,
              y: d[line.dataKey] as number | null | undefined,
            }));

            const path = generateLinePath(lineData, xScale, yScaleFunc, line.smooth);

            return (
              <path
                id={`line-${i}-${line.dataKey}`}
                key={`line-${i}-${line.dataKey}`}
                d={path}
                fill="none"
                stroke={line.stroke}
                strokeWidth={line.strokeWidth || 2}
                strokeOpacity={line.strokeOpacity || 1}
              />
            );
          })}

          {customElementGroup}

          {/* Synchronized crosshair */}
          {hoveredX !== null && (
            <line
              x1={xScale(hoveredX)}
              y1={0}
              x2={xScale(hoveredX)}
              y2={chartHeight}
              stroke="#999"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.6}
              pointerEvents="none"
            />
          )}

          {/* Reference points with labels - auto-generated from x-ticks */}
          {showReferencePoints && xTicks.map((xTickValue, i) => {
            const xPos = xScale(xTickValue);

            // Find the main line data (first line with non-zero strokeWidth or last line)
            const mainLine = lines.find(line => (line.strokeWidth || 2) > 1) || lines[lines.length - 1];
            if (!mainLine) return null;
            const mainColor = mainLine.stroke;

            // Find the y value at this x-tick position from the data
            const lineData = data.filter(d => {
              const val = d[mainLine.dataKey];
              return val != null && !isNaN(val as number);
            });

            if (lineData.length === 0) return null;

            // Find closest data point to this x-tick
            const closestPoint = lineData.reduce((prev, curr) => {
              return Math.abs(curr.x - xTickValue) < Math.abs(prev.x - xTickValue) ? curr : prev;
            }, lineData[0]);

            const yValue = closestPoint[mainLine.dataKey] as number;
            if (!Number.isFinite(yValue)) return null;

            const closestPointIndex = lineData.indexOf(closestPoint);
            const yPos = yScaleFunc(yValue);

            // Skip if yPos is NaN or invalid
            if (!Number.isFinite(yPos)) return null;

            // Calculate clamping state
            const isClampedAbove = yPos < 0;
            const isClampedBelow = yPos > chartHeight;
            const isClamped = isClampedAbove || isClampedBelow;
            const clampedYPos = clamp(yPos, 0, chartHeight);

            // Check if point is within 10% of the maximum value (top of chart)
            const isNearMaximum = yPos <= chartHeight * 0.1 && yPos >= 0;

            // Check if point is within 10% of the minimum value (bottom of chart)
            const isNearMinimum = yPos >= chartHeight * 0.9 && yPos <= chartHeight;

            // Points that are off the TOP or near maximum should have labels below
            // Points that are off the BOTTOM or near minimum should have labels above
            const shouldPositionBelow = isClampedAbove || isNearMaximum;
            const shouldPositionAbove = isNearMinimum || isClampedBelow;

            // Calculate smart offset based on line normal
            let labelOffsetX = 8;
            let labelOffsetY = 4;

            if (shouldPositionBelow) {
              labelOffsetX = 0;
              labelOffsetY = 16;
            } else if (shouldPositionAbove) {
              labelOffsetX = 0;
              labelOffsetY = -12;
            } else if (lineData.length > 1 && closestPointIndex >= 0) {
              // Find neighboring points for slope calculation
              const prevPoint = closestPointIndex > 0 ? lineData[closestPointIndex - 1] : lineData[0];
              const nextPoint = closestPointIndex < lineData.length - 1 ? lineData[closestPointIndex + 1] : lineData[lineData.length - 1];

              const prevY = prevPoint[mainLine.dataKey] as number;
              const nextY = nextPoint[mainLine.dataKey] as number;

              if (prevY != null && nextY != null && prevPoint !== nextPoint) {
                const dx = xScale(nextPoint.x) - xScale(prevPoint.x);
                const dy = yScaleFunc(nextY) - yScaleFunc(prevY);

                const length = Math.sqrt(dx * dx + dy * dy) || 1;
                const normalX = -dy / length;
                const normalY = dx / length;

                const centerX = chartWidth / 2;
                const centerY = chartHeight / 2;
                const toCenterX = centerX - xPos;
                const toCenterY = centerY - clampedYPos;

                const dotProduct = normalX * toCenterX + normalY * toCenterY;
                const direction = dotProduct >= 0 ? 1 : -1;

                const offsetDistance = 8;
                labelOffsetX = direction * normalX * offsetDistance;
                labelOffsetY = direction * normalY * offsetDistance;

                const centerBias = 0.2;
                const centerDist = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY) || 1;
                labelOffsetX += (toCenterX / centerDist) * offsetDistance * centerBias;
                labelOffsetY += (toCenterY / centerDist) * offsetDistance * centerBias;
              }
            }

            // Format label using yTickFormatter if available
            const labelContent = yTickFormatter ? yTickFormatter(yValue, true) : yValue.toString();
            const isReactNode = typeof labelContent !== 'string';

            return (
              <g id={`ref-point-${i}`} key={`ref-point-${i}`}>
                {/* Marker circle removed to avoid overlap with x-axis tick points */}

                {/* Directional arrow indicator for out-of-bounds values */}
                {isClampedAbove && (
                  <polygon
                    points={`${xPos-4},${clampedYPos-8} ${xPos+4},${clampedYPos-8} ${xPos},${clampedYPos-2}`}
                    fill={mainColor}
                    opacity={0.7}
                  />
                )}

                {isClampedBelow && (
                  <polygon
                    points={`${xPos-4},${clampedYPos+8} ${xPos+4},${clampedYPos+8} ${xPos},${clampedYPos+2}`}
                    fill={mainColor}
                    opacity={0.7}
                  />
                )}

                {/* Label next to the marker */}
                {isReactNode ? (
                  <foreignObject
                    x={xPos + labelOffsetX - ((shouldPositionBelow || shouldPositionAbove) ? 50 : 0)}
                    y={clampedYPos + labelOffsetY - 8}
                    width={100}
                    height={20}
                    style={{ overflow: 'visible' }}
                  >
                    <div style={{
                      backgroundColor: 'red',
                      fontSize: '11px',
                      color: mainColor,
                      fontWeight: '500',
                      fontFamily: 'inherit',
                      textAlign: (shouldPositionBelow || shouldPositionAbove) ? 'center' : 'left'
                    }}>
                      {isClamped && (isClampedAbove ? '↑ ' : '↓ ')}
                      {labelContent}
                    </div>
                  </foreignObject>
                ) : (
                  <text
                    x={xPos + labelOffsetX}
                    y={clampedYPos + labelOffsetY}
                    textAnchor={(shouldPositionBelow || shouldPositionAbove) ? "middle" : (labelOffsetX > 0 ? "start" : "end")}
                    fontSize={11}
                    fill={mainColor}
                    fontWeight="500"
                  >
                    {isClamped && (isClampedAbove ? '↑ ' : '↓ ')}
                    {labelContent}
                  </text>
                )}
              </g>
            );
          })}

          {/* Y Axis */}
          {showYAxis && yTicks && yTicks.length > 0 && (
            <g>
              {yTicks.map((tick, i) => {
                const yPos = yScaleFunc(tick);
                const labelContent = yTickFormatter ? yTickFormatter(tick) : tick.toString();
                const isReactNode = typeof labelContent !== 'string';
                return (
                  <g key={`y-tick-${i}`} transform={`translate(0,${yPos})`}>
                    {isReactNode ? (
                      <foreignObject x={10} y={-10} width={80} height={20} style={{ overflow: 'visible' }}>
                        <div style={{ fontSize: '11px', color: '#666', fontWeight: 500, fontFamily: 'inherit' }}>
                          {labelContent as unknown as ReactNode}
                        </div>
                      </foreignObject>
                    ) : (
                      <text
                        x={10}
                        textAnchor="start"
                        alignmentBaseline="middle"
                        fontSize={11}
                        fill="#666"
                      >
                        {labelContent as string}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* X Axis */}
          {showXAxis && (
          <g transform={`translate(0,${chartHeight})`}>
            {xTicks.map((tick, i) => (
              <g key={`x-tick-${i}`} transform={`translate(${xScale(tick)},0)`}>
                <text
                  y={20}
                  textAnchor="middle"
                  fontSize={12}
                  fill="#666"
                >
                  {xTickFormatter(tick)}
                </text>
              </g>
            ))}
            {xLabel && (
              <text
                x={chartWidth / 2}
                y={50}
                textAnchor="middle"
                fontSize={14}
                fill="#666"
              >
                {xLabel}
                </text>
              )}
            </g>
          )}

        </g>
      </svg>

      {/* Tooltip */}
      {tooltip && hoveredPoint && mousePosition && (
        <div
          style={{
            position: 'fixed',
            left: mousePosition.x + 12,
            top: mousePosition.y + 12,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          {tooltip(hoveredPoint)}
        </div>
      )}
    </div>
  );
}
