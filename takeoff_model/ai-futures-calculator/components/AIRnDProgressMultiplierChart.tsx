'use client';

import { memo, useMemo, useCallback } from 'react';
import { CustomLineChart } from './CustomLineChart';
import type { ChartDataPoint } from '@/app/types';
import type { DataPoint } from './CustomLineChart';
import { formatAsPowerOfTenText, formatYearMonthShort } from '@/utils/formatting';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import type { MilestoneMap } from '@/types/milestones';
import { createScale, calculateDynamicXDomain, clamp } from '@/utils/chartUtils';
import { AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION } from '@/constants/chartExplanations';
import { WithChartTooltip } from './ChartTitleTooltip';

const MINIMUM_VISIBLE_YEAR = 2019;

export interface AIRnDProgressMultiplierChartProps {
  chartData: ChartDataPoint[];
  tooltip: (point: DataPoint) => React.ReactNode;
  milestones: MilestoneMap | null;
  displayEndYear: number;
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  className?: string;
}

const milestoneSequence: Array<{ key: string; label: string }> = [
  { key: 'AC', label: 'AC' },
  { key: 'SAR-level-experiment-selection-skill', label: 'SAR' },
  { key: 'SIAR-level-experiment-selection-skill', label: 'SIAR' },
  // { key: 'STRAT-AI', label: 'STRAT-AI' },
  { key: 'TED-AI', label: 'TED-AI' },
  { key: 'ASI', label: 'ASI' },
];

export const AIRnDProgressMultiplierChart = memo(({
  chartData,
  tooltip,
  milestones,
  displayEndYear,
  verticalReferenceLine,
  className,
}: AIRnDProgressMultiplierChartProps) => {
  const seriesPoints = useMemo(() => {
    return chartData
      .filter(point =>
        typeof point.year === 'number' &&
        Number.isFinite(point.year) &&
        typeof point.aiSwProgressMultRefPresentDay === 'number' &&
        Number.isFinite(point.aiSwProgressMultRefPresentDay as number)
      )
      .map(point => ({
        x: point.year as number,
        y: point.aiSwProgressMultRefPresentDay as number,
      }))
      .sort((a, b) => a.x - b.x);
  }, [chartData]);

  const milestoneDomain = useMemo<[number, number] | undefined>(() => {
    if (!milestones || seriesPoints.length === 0) {
      return undefined;
    }

    // Find all achieved milestones (those with valid time values) from the sequence
    const achievedTimes = milestoneSequence
      .map(({ key }) => milestones[key]?.time)
      .filter((time): time is number => typeof time === 'number' && Number.isFinite(time));

    // Need at least 2 milestones (including AC) to create a meaningful domain
    if (achievedTimes.length < 2) {
      return undefined;
    }

    const start = Math.min(...achievedTimes);
    const end = Math.max(...achievedTimes);

    // Find the first data point >= end to ensure we include milestone values in y-domain
    const nextDataPoint = seriesPoints.find(p => p.x >= end);
    const effectiveEnd = nextDataPoint ? nextDataPoint.x : end;

    const range = effectiveEnd - start;
    const padding = range * 0.04;
    const clampedStart = Math.max(start - padding, MINIMUM_VISIBLE_YEAR);
    const clampedEnd = effectiveEnd + padding;

    return [clampedStart, clampedEnd];
  }, [milestones, seriesPoints]);

  const xDomain = useMemo<[number, number]>(() => {
    if (milestoneDomain) {
      return milestoneDomain;
    }

    if (seriesPoints.length === 0) {
      return [2020, 2030];
    }

    const years = seriesPoints.map(point => point.x);
    const intersectionX = verticalReferenceLine?.x ?? null;

    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.1,
      maxPadding: 0.15,
      minRange: 3,
      maxDomain: displayEndYear,
    });
  }, [seriesPoints, milestoneDomain, verticalReferenceLine, displayEndYear]);

  const xTickFormatter = useCallback((value: number) => formatYearMonthShort(value), []);

  const milestoneMarkers = useMemo(() => {
    if (!milestones) {
      return [] as Array<{ x: number; label: string; value: number }>;
    }

    return milestoneSequence
      .map(({ key, label }) => {
        const rawTime = milestones[key]?.time;
        const year = typeof rawTime === 'number' && Number.isFinite(rawTime) ? rawTime : null;
        const value = milestones[key]?.progress_multiplier;
        return {
          x: year ?? NaN,
          label,
          value,
        };
      })
      .filter(marker => Number.isFinite(marker.x) && marker.value != null && Number.isFinite(marker.value) && (marker.value as number) > 0)
      .map(marker => ({ x: marker.x as number, label: marker.label, value: marker.value as number }));
  }, [milestones]);

  const yDomain = useMemo<[number, number]>(() => {
    if (seriesPoints.length === 0) {
      return [0.1, 10];
    }

    const [domainStart, domainEnd] = xDomain;

    const inDomain = seriesPoints.filter(point => point.x >= domainStart && point.x <= domainEnd && Number.isFinite(point.y) && point.y > 0);

    // Include milestone values in the y-domain calculation
    const milestoneValues = milestoneMarkers
      .filter(m => m.x >= domainStart && m.x <= domainEnd)
      .map(m => m.value);

    const allValues = [...inDomain.map(point => point.y), ...milestoneValues];

    if (allValues.length === 0) {
      return [0.1, 10];
    }

    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue <= 0) {
      return [0.1, 10];
    }

    const lower = Math.max(minValue * 0.7, Number.MIN_VALUE);
    let upper = Math.max(maxValue * 2, lower * 1.1);

    if (!Number.isFinite(upper) || upper <= lower) {
      upper = lower * 10;
    }

    return [lower, upper];
  }, [seriesPoints, xDomain, milestoneMarkers]);

  const findMilestoneLabel = useCallback((year: number): string | undefined => {
    let closestLabel: string | undefined;
    let smallestDiff = Number.POSITIVE_INFINITY;

    milestoneMarkers.forEach(marker => {
      const diff = Math.abs(marker.x - year);
      if (diff < smallestDiff && diff <= 0.1) {
        smallestDiff = diff;
        closestLabel = marker.label;
      }
    });

    return closestLabel;
  }, [milestoneMarkers]);

  const mergedData = useMemo(() => {
    if (seriesPoints.length === 0) {
      return chartData.map(point => ({ ...point, x: point.year }));
    }

    const seriesLookup = new Map(seriesPoints.map(point => [point.x, point.y]));

    return chartData
      .filter(point => typeof point.year === 'number' && Number.isFinite(point.year))
      .map(point => ({
        ...point,
        x: point.year,
        aiSwProgressMultRefPresentDay: seriesLookup.get(point.year as number) ?? point.aiSwProgressMultRefPresentDay,
        milestoneLabel: findMilestoneLabel(point.year as number),
      }));
  }, [chartData, seriesPoints, findMilestoneLabel]);

  const customElements = useMemo(() => {
    if (milestoneMarkers.length === 0) {
      return undefined;
    }

    const MilestoneOverlay = (context: { chartWidth: number; chartHeight: number, xScale: (value: number) => number, yScale: (value: number) => number }) => {
      const { chartWidth, chartHeight } = context;
      const { xDomain: renderedXDomain, yDomain: renderedYDomain, xScale, yScale } = context as typeof context & {
        xDomain: [number, number];
        yDomain: [number, number];
      };

      return (
        <g>
          {milestoneMarkers.map((marker, index) => {
            const xPos = xScale(marker.x);
            const yPos = yScale(marker.value);

            if (!Number.isFinite(xPos) || !Number.isFinite(yPos)) {
              return null;
            }

            const clampedY = clamp(yPos, 0, chartHeight);
            const labelYOffset = -12;
            const horizontalPadding = 12;

            const nearLeftEdge = xPos <= horizontalPadding;
            const nearRightEdge = xPos >= chartWidth - horizontalPadding;

            let textAnchor: 'start' | 'middle' | 'end' = 'end';
            let textX = clamp(xPos, horizontalPadding, chartWidth - horizontalPadding);

            if (nearLeftEdge) {
              textAnchor = 'start';
              textX = Math.max(xPos + horizontalPadding * 0.5, horizontalPadding);
            } else if (nearRightEdge) {
              textAnchor = 'end';
              textX = Math.min(xPos - horizontalPadding * 0.5, chartWidth - horizontalPadding);
            }

            const textY = clamp(clampedY + labelYOffset, 12, chartHeight - 4);

            return (
              <g key={`milestone-${index}`}>
                <circle
                  cx={xPos}
                  cy={clampedY}
                  r={4}
                  fill="#2A623D"
                  stroke="white"
                  strokeWidth={1.5}
                />
                <text
                  x={textX}
                  y={textY}
                  textAnchor={textAnchor}
                  fontSize={11}
                  fill="#2A623D"
                  fontWeight="600"
                >
                  {marker.label}
                </text>
              </g>
            );
          })}
        </g>
      );
    };

    MilestoneOverlay.displayName = 'AIRnDMilestoneOverlay';

    return MilestoneOverlay;
  }, [milestoneMarkers]);

  const lines = useMemo(() => [
    {
      dataKey: 'aiSwProgressMultRefPresentDay',
      stroke: '#2A623D',
      strokeWidth: 3,
      strokeOpacity: 1,
      name: 'AI Software R&D Uplift',
    },
  ], []);

  return (
    <div className={className ? `flex flex-col ${className}` : 'flex flex-col'} style={{ position: 'relative' }}>
      <div className="flex gap-2 items-center">
        <span className="text-[12px] font-semibold text-primary text-left font-system-mono">
          Takeoff Period: AI Software R&amp;D Uplift
        </span>
        <div className="flex-1 border-t border-gray-500/30" />
        <WithChartTooltip explanation={AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION} className="!gap-0"><></></WithChartTooltip>
      </div>
      <CustomLineChart
        data={mergedData}
        height={CHART_LAYOUT.primary.height}
        width={CHART_LAYOUT.primary.width}
        margin={{ top: 0, right: 60, left: 0, bottom: 35 }}
        xPadding={{ start: 16 }}
        xDomain={xDomain}
        xTickCount={4}
        xTickFormatter={xTickFormatter}
        yDomain={yDomain}
        yScale="log"
        yTickFormatter={(value) => formatAsPowerOfTenText(value, { suffix: ' x' })}
        showReferencePoints={false}
        lines={lines}
        tooltip={tooltip}
        customElements={customElements}
        showXAxis
      />
    </div>
  );
});

AIRnDProgressMultiplierChart.displayName = 'AIRnDProgressMultiplierChart';


