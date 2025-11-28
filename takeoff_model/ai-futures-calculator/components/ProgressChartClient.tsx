'use client';

import { useState, useCallback, useEffect, useRef, useMemo, ChangeEvent } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import CustomHorizonChart from './CustomHorizonChart';
import CombinedComputeChart from './CombinedComputeChart';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import { CustomMetricChart } from './CustomMetricChart';
import type { DataPoint } from './CustomLineChart';

import { formatTo3SigFigs, formatTimeDuration, formatSCHorizon, yearsToMinutes, formatYearMonth, formatAsPowerOfTenNode, formatAsPowerOfTenText, formatWorkTimeDuration, formatWorkTimeDurationDetailed } from '@/utils/formatting';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { convertParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ParameterSlider } from './ParameterSlider';
import { AdvancedSections } from './AdvancedSections';
import { ChartType } from '@/types/chartConfig';
import { CHART_CONFIGS } from '@/utils/chartConfigs';
import { ChartSyncProvider } from './ChartSyncContext';
import { ChartLoadingOverlay } from './ChartLoadingOverlay';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES } from '@/utils/urlState';
import ModelArchitecture from './ModelArchitecture';
import { AIRnDProgressMultiplierChart } from './AIRnDProgressMultiplierChart';
import type { MilestoneMap } from '@/types/milestones';
import { SmallChartMetricTooltip } from './SmallChartMetricTooltip';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';
import { ParameterHoverProvider } from './ParameterHoverContext';
import { HeaderContent } from './HeaderContent';
import { ModelFeatureCheckbox } from './ModelFeatureCheckbox';
import { WithChartTooltip } from './ChartTitleTooltip';

interface ModelDefaults {
  optimal_ces_eta_init?: number;
  automation_interp_type?: string;
  ai_research_taste_slope?: number;
  anchor_progress_at_strong_cognitive_horizon?: number;
  present_year?: number;
  present_progress?: number;
  progress_at_aa?: number;
  [key: string]: number | string | undefined;
}

interface ParameterConfig {
  defaults?: ModelDefaults;
  bounds?: Record<string, [number, number]>;
  metadata?: Record<string, unknown>;
}

interface TimeSeriesPoint {
  year: number;
  progress: number;
  effectiveCompute: number;
  horizonLength: number;
  researchStock: number;
  automationFraction?: number;
  trainingCompute?: number;
  experimentCapacity?: number;
  aiResearchTaste?: number;
  aiSoftwareProgressMultiplier?: number;
  aiSwProgressMultRefPresentDay?: number;
  serialCodingLaborMultiplier?: number;
  // Newly exposed metrics from API
  humanLabor?: number;
  inferenceCompute?: number;
  experimentCompute?: number;
  researchEffort?: number;
  softwareProgressRate?: number;
  softwareEfficiency?: number;
  aiCodingLaborMultiplier?: number;
  aiCodingLaborMultRefPresentDay?: number;
}

const CHART_ANIMATION_DURATION_MS = 450;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const interpolateNumericValue = (
  startValue: unknown,
  endValue: unknown,
  progress: number
): number | null => {
  const startNumeric = isFiniteNumber(startValue)
    ? startValue
    : isFiniteNumber(endValue)
      ? endValue
      : null;
  const endNumeric = isFiniteNumber(endValue) ? endValue : startNumeric;

  if (startNumeric == null || endNumeric == null) {
    return null;
  }

  if (progress <= 0) {
    return startNumeric;
  }

  if (progress >= 1) {
    return endNumeric;
  }

  return startNumeric + (endNumeric - startNumeric) * progress;
};

const interpolateChartDataPoints = (
  fromData: ChartDataPoint[],
  toData: ChartDataPoint[],
  progress: number
): ChartDataPoint[] => {
  if (progress >= 1) {
    return toData.map(point => ({ ...point }));
  }

  if (fromData.length === 0) {
    return toData.map(point => ({ ...point }));
  }

  const fromMap = new Map<number, ChartDataPoint>();
  const toMap = new Map<number, ChartDataPoint>();

  fromData.forEach(point => {
    fromMap.set(point.year, point);
  });
  toData.forEach(point => {
    toMap.set(point.year, point);
  });

  const years = Array.from(new Set([...fromMap.keys(), ...toMap.keys()])).sort(
    (a, b) => a - b
  );

  return years.map(year => {
    const startPoint = fromMap.get(year);
    const endPoint = toMap.get(year);

    if (!startPoint && !endPoint) {
      return {
        year,
        horizonLength: 0,
        horizonFormatted: '',
      };
    }

    const resolvedStart = startPoint ?? endPoint!;
    const resolvedEnd = endPoint ?? startPoint!;
    const mergedKeys = new Set([
      ...Object.keys(resolvedStart),
      ...Object.keys(resolvedEnd),
    ]);

    const interpolatedHorizon =
      interpolateNumericValue(
        resolvedStart.horizonLength,
        resolvedEnd.horizonLength,
        progress
      ) ??
      (typeof resolvedEnd.horizonLength === 'number'
        ? resolvedEnd.horizonLength
        : typeof resolvedStart.horizonLength === 'number'
          ? resolvedStart.horizonLength
          : 0);

    const interpolatedPoint: ChartDataPoint = {
      year,
      horizonLength: interpolatedHorizon,
      horizonFormatted:
        progress >= 1
          ? resolvedEnd.horizonFormatted ?? resolvedStart.horizonFormatted ?? ''
          : resolvedStart.horizonFormatted ?? resolvedEnd.horizonFormatted ?? '',
    };

    mergedKeys.forEach(key => {
      if (key === 'year' || key === 'horizonLength' || key === 'horizonFormatted') {
        return;
      }

      const startValue = resolvedStart[key];
      const endValue = resolvedEnd[key];

      if (isFiniteNumber(startValue) || isFiniteNumber(endValue)) {
        interpolatedPoint[key] = interpolateNumericValue(startValue, endValue, progress);
      } else {
        interpolatedPoint[key] = progress >= 1
          ? endValue ?? startValue ?? null
          : startValue ?? endValue ?? null;
      }
    });

    return interpolatedPoint;
  });
};

const VISIBLE_CHART_START_YEAR = 2019;
const SIMULATION_END_YEAR = 2045; // Simulation end year (matches sampling_config.yaml)
const CODING_AUTOMATION_MARKER_COLOR = '#6b7280';
// Benchmark data will be loaded dynamically from YAML file

type MilestoneDisplay = {
  month: string;
  monthNumber: number;
  year: string;
};

type ChartListItem = {
  id: string;
  title: string;
};

const getMilestoneDisplay = (decimalYear: number | null): MilestoneDisplay | null => {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);

  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  return {
    month: months[monthIndex],
    monthNumber: monthIndex + 1,
    year: String(adjustedYear),
  };
};

interface KeyMilestoneValueProps {
  label: string;
  milestone: MilestoneDisplay | null;
  fallback: string;
  isLoading: boolean;
}

const KeyMilestoneValue = ({
  label,
  milestone,
  fallback,
  isLoading,
}: KeyMilestoneValueProps) => (
  <div>
    <div className="flex items-center gap-2">
      <p className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">{label}</p>
      <div className="flex-1 border-t border-gray-500/30" />
    </div>
    <div className="mt-1 min-h-[3.5rem]">
      {isLoading ? (
        <div className="h-12 w-2/3 animate-pulse rounded bg-slate-200/70" />
      ) : milestone ? (
        <>
          <div className="text-5xl font-et-book text-primary mb-0 leading-none flex justify-between w-full">
            {`${milestone.monthNumber}/${milestone.year}`.split('').map((char, idx) => (
              <span key={idx}>{char}</span>
            ))}
          </div>
        </>
      ) : (
        <div className="text-xl font-semibold text-slate-500">{fallback}</div>
      )}
    </div>
  </div>
);

interface KeyMilestonePanelProps {
  automationYear: number | null;
  asiYear: number | null;
  isLoading: boolean;
  primaryCharts: ChartListItem[];
  metricCharts: ChartListItem[];
  hiddenCharts: Set<string>;
  onToggleChart: (chartId: string) => void;
}

const KeyMilestonePanel = ({
  automationYear,
  asiYear,
  isLoading,
  primaryCharts,
  metricCharts,
  hiddenCharts,
  onToggleChart,
}: KeyMilestonePanelProps) => {
  const automationMilestone = getMilestoneDisplay(automationYear);
  const asiMilestone = getMilestoneDisplay(asiYear);
  const renderChartList = (charts: ChartListItem[]) => (
    charts.map((chart) => {
      const isHidden = hiddenCharts.has(chart.id);
      return (
        <button
          key={chart.id}
          type="button"
          onClick={() => onToggleChart(chart.id)}
          aria-pressed={!isHidden}
          className={`group flex items-center gap-1 rounded-sm pl-[1px] text-left transition ${isHidden ? 'opacity-45' : 'opacity-100 hover:bg-slate-100/70'}`}
        >
          <span className="font-sans uppercase text-[9px] font-semibold tracking-[0.08em] whitespace-nowrap overflow-hidden text-ellipsis">
            {chart.title}
          </span>
          <span
            className={`ml-auto h-1.5 w-1.5 min-w-1.5 min-h-1.5 rounded-full transition ${isHidden ? 'bg-slate-300' : 'bg-accent group-hover:scale-125'}`}
          />
        </button>
      );
    })
  );

  return (
    <section
      className="flex h-full flex-col gap-4"
      style={{
        minWidth: CHART_LAYOUT.keyStats.width,
        maxWidth: CHART_LAYOUT.keyStats.width,
        minHeight: CHART_LAYOUT.primary.height,
      }}
    >
      <div className="flex flex-col gap-0">
        <KeyMilestoneValue
          label="Date of Automated Coder"
          milestone={automationMilestone}
          fallback="Beyond 2045"
          isLoading={isLoading}
        />
        <KeyMilestoneValue
          label="Date of ASI"
          milestone={asiMilestone}
          fallback="Not reached"
          isLoading={isLoading}
        />
        <div className="flex gap-2 items-center my-1">
          <span className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">Charts</span>
          <div className="flex-1 border-t border-gray-500/30" />
        </div>
        <div className="flex flex-col gap-[2.5px]">
          {primaryCharts.length > 0 && renderChartList(primaryCharts)}
          {primaryCharts.length > 0 && metricCharts.length > 0 && (
            <div className="border-t border-gray-500/20 my-0.5" />
          )}
          {metricCharts.length > 0 && renderChartList(metricCharts)}
        </div>
      </div>
    </section>
  );
};

const PARAMETER_PRESETS = [
  { id: 'baseline', label: 'Baseline [not yet functional]' },
  { id: 'automation-forward', label: 'Alternative [not yet functional]' },
//  { id: 'compute-heavy', label: 'Compute-Heavy' },
];

const HEADER_REVEAL_OFFSET = 120;
const SCROLL_DIRECTION_THRESHOLD = 6;

// Custom hook to sync checkbox state with parameter values
interface CheckboxParameterSyncConfig {
  checkboxValue: boolean;
  checkboxKey: keyof typeof DEFAULT_CHECKBOX_STATES;
  nonDefaultParameterValues: Partial<ParametersType>;
  setParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
  setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
  isHydratingFromUrlRef: React.MutableRefObject<boolean>;
}

function useCheckboxParameterSync({
  checkboxValue,
  checkboxKey,
  nonDefaultParameterValues,
  setParameters,
  setUiParameters,
  isHydratingFromUrlRef,
}: CheckboxParameterSyncConfig) {
  useEffect(() => {
    if (isHydratingFromUrlRef.current) {
      return;
    }

    const isDefaultState = checkboxValue === DEFAULT_CHECKBOX_STATES[checkboxKey];
    const defaultCheckboxValue = DEFAULT_CHECKBOX_STATES[checkboxKey];
    const isInNonDefaultState = checkboxValue !== defaultCheckboxValue;

    const nonDefaultUpdaterFn = (prev: ParametersType) => ({
      ...prev,
      ...nonDefaultParameterValues,
      // We need the cast because in theory we could be getting an object with explicit undefined values for some keys
      // Hopefully nobody does that...
    } as ParametersType);

    // When in NON-DEFAULT state (unchecked), apply the special parameter values
    // When in DEFAULT state (checked), apply the default parameter values
    if (isInNonDefaultState) {
      // Apply non-default parameter values when checkbox is unchecked
      setUiParameters(nonDefaultUpdaterFn);
      setParameters(nonDefaultUpdaterFn);
    } else if (isDefaultState) {
      const parameterKeys = Object.keys(nonDefaultParameterValues) as Array<keyof ParametersType>;
      const defaultValues: Partial<ParametersType> = {};
      parameterKeys.forEach(key => {
        defaultValues[key] = DEFAULT_PARAMETERS[key];
      });

      const defaultUpdaterFn = (prev: ParametersType) => ({
        ...prev,
        ...defaultValues,
      } as ParametersType);

      setUiParameters(defaultUpdaterFn);
      setParameters(defaultUpdaterFn);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkboxValue]);
}

// Types
interface ProgressChartProps {
  benchmarkData?: BenchmarkPoint[];
  IntroductionMarkdown?: React.ReactNode;
}

export default function ProgressChart({ benchmarkData = [] }: ProgressChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [animatedChartData, setAnimatedChartData] = useState<ChartDataPoint[]>([]);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, DEFAULT_PARAMETERS.ac_time_horizon_minutes)
  );
  const [activePopover, setActivePopover] = useState<ChartType | null>(null);
  const [parameters, setParameters] = useState<ParametersType>({ ...DEFAULT_PARAMETERS });
  // Separate UI parameters for immediate slider updates vs committed parameters for API calls
  const [uiParameters, setUiParameters] = useState<ParametersType>({ ...DEFAULT_PARAMETERS });
  const [isDragging, setIsDragging] = useState(false);
  const [allParameters, setAllParameters] = useState<ParameterConfig | null>(null);
  const [trajectoryData, setTrajectoryData] = useState<ChartDataPoint[]>([]);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(null);
  const [mainLoading, setMainLoading] = useState(false);
  const [isParameterConfigReady, setIsParameterConfigReady] = useState(false);
  const [isAdvancedPanelOpen, setIsAdvancedPanelOpen] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const [horizonParams, setHorizonParams] = useState<{ uses_shifted_form: boolean; anchor_progress: number | null } | null>(null);
  const [summary, setSummary] = useState<{ beta_software?: number; r_software?: number;[key: string]: unknown } | null>(null);
  const [hiddenCharts, setHiddenCharts] = useState<Set<string>>(new Set());
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isUrlSyncReady, setIsUrlSyncReady] = useState(false);

  const [enableCodingAutomation, setEnableCodingAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableCodingAutomation);
  const [enableExperimentAutomation, setEnableExperimentAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
  const [useExperimentThroughputCES, setUseExperimentThroughputCES] = useState(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
  const [enableSoftwareProgress, setEnableSoftwareProgress] = useState(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
  const [useComputeLaborGrowthSlowdown, setUseComputeLaborGrowthSlowdown] = useState(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
  const [useVariableHorizonDifficulty, setUseVariableHorizonDifficulty] = useState(DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty);

  // Flag to prevent checkbox effects during initial URL hydration
  const isHydratingFromUrlRef = useRef(false);

  const lockedParameters = useMemo(() => {
    const locked = new Set<string>();
    // Lock when unchecked (disabled state)
    if (!enableCodingAutomation) {
      locked.add('optimal_ces_eta_init');
      locked.add('coding_automation_efficiency_slope');
      locked.add('swe_multiplier_at_present_day');
    }
    if (!enableExperimentAutomation) {
      locked.add('median_to_top_taste_multiplier');
    }
    // Lock when unchecked (uses Cobb-Douglas with high asymptotes)
    if (!useExperimentThroughputCES) {
      locked.add('direct_input_exp_cap_ces_params');
      locked.add('inf_labor_asymptote');
      locked.add('inf_compute_asymptote');
      locked.add('inv_compute_anchor_exp_cap');
      locked.add('rho_experiment_capacity');
    }
    if (!enableSoftwareProgress) {
      locked.add('software_progress_rate_at_reference_year');
    }
    // Lock when unchecked (sets to constant=1)
    if (!useVariableHorizonDifficulty) {
      locked.add('doubling_difficulty_growth_factor');
    }
    return locked;
  }, [enableCodingAutomation, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useVariableHorizonDifficulty]);

  const scHorizonLogBounds = useMemo(() => {
    const rawBounds = allParameters?.bounds?.ac_time_horizon_minutes;
    if (rawBounds && rawBounds[0] > 0 && rawBounds[1] > 0) {
      return {
        min: Math.log10(rawBounds[0]),
        max: Math.log10(rawBounds[1])
      };
    }
    return { min: 3, max: 11 };
  }, [allParameters]);

  const preGapHorizonBounds = useMemo(() => {
    const rawBounds = allParameters?.bounds?.pre_gap_ac_time_horizon;
    if (rawBounds && rawBounds[0] > 0 && rawBounds[1] > 0) {
      return {
        min: rawBounds[0],
        max: rawBounds[1]
      };
    }
    return { min: 1000, max: 100000000000 };
  }, [allParameters]);

  const parallelPenaltyBounds = useMemo(() => {
    const rawBounds = allParameters?.bounds?.parallel_penalty;
    if (rawBounds) {
      return {
        min: rawBounds[0],
        max: rawBounds[1]
      };
    }
    return { min: 0, max: 1 };
  }, [allParameters]);

  // Ref for debouncing Monte Carlo updates
  const monteCarloTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Ref for request deduplication
  const currentMonteCarloRequestRef = useRef<AbortController | null>(null);
  // Ref to cancel in-flight chart computations when parameters change quickly
  const chartRequestControllerRef = useRef<AbortController | null>(null);
  const latestChartRequestIdRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const animationStartRef = useRef<number>(0);
  const animationFromRef = useRef<ChartDataPoint[]>([]);
  const animationToRef = useRef<ChartDataPoint[]>([]);
  const animatedDataRef = useRef<ChartDataPoint[]>([]);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const lastScrollTopRef = useRef(0);

  const commitAnimatedChartData = useCallback((nextData: ChartDataPoint[]) => {
    animatedDataRef.current = nextData;
    setAnimatedChartData(nextData);
  }, []);

  const cancelChartAnimation = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const toggleChartVisibility = useCallback((chartId: string) => {
    setHiddenCharts((prev) => {
      const next = new Set(prev);
      if (next.has(chartId)) {
        next.delete(chartId);
      } else {
        next.add(chartId);
      }
      return next;
    });
  }, []);

  const startChartAnimation = useCallback(
    (nextData: ChartDataPoint[], options: { immediate?: boolean } = {}) => {
      const shouldSkipAnimation =
        options.immediate ?? animatedDataRef.current.length === 0;

      if (shouldSkipAnimation) {
        cancelChartAnimation();
        animationFromRef.current = nextData;
        animationToRef.current = nextData;
        animationStartRef.current = 0;
        commitAnimatedChartData(nextData);
        return;
      }

      animationFromRef.current = animatedDataRef.current;
      animationToRef.current = nextData;
      animationStartRef.current = performance.now();
      cancelChartAnimation();

      const step = (timestamp: number) => {
        const elapsed = timestamp - animationStartRef.current;
        const progress = Math.min(1, elapsed / CHART_ANIMATION_DURATION_MS);
        const interpolated = interpolateChartDataPoints(
          animationFromRef.current,
          animationToRef.current,
          progress
        );
        commitAnimatedChartData(interpolated);

        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(step);
        } else {
          animationFrameRef.current = null;
        }
      };

      animationFrameRef.current = requestAnimationFrame(step);
    },
    [cancelChartAnimation, commitAnimatedChartData]
  );

  useEffect(() => {
    return () => {
      cancelChartAnimation();
    };
  }, [cancelChartAnimation]);

  // Memoized popover handlers to prevent chart re-renders
  const handlePopoverOpen = useCallback((type: ChartType) => {
    setActivePopover(type);
  }, []);

  const handlePopoverClose = useCallback(() => {
    setActivePopover(null);
  }, []);

  const handleExportParameters = useCallback(() => {
    const parameterData = JSON.stringify(uiParameters, null, 2);
    const blob = new Blob([parameterData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'parameter-settings.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [uiParameters]);

  const handleImportParameters = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const fileContents = await file.text();
      const parsed = JSON.parse(fileContents);
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Invalid parameter file');
      }

      const parsedParameters = parsed as ParametersType;
      setUiParameters(parsedParameters);
      setParameters(parsedParameters);
      setIsDragging(false);
    } catch (error) {
      console.error('Failed to import parameters:', error);
    } finally {
      event.target.value = '';
    }
  }, [setUiParameters, setParameters, setIsDragging]);

  const handleImportButtonClick = useCallback(() => {
    importInputRef.current?.click();
  }, []);

  const handlePresetSelect = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    setSelectedPresetId(event.target.value);
  }, []);

  const handleResetKeyParameters = useCallback(() => {
    const defaults = { ...DEFAULT_PARAMETERS };
    setUiParameters(defaults);
    setParameters(defaults);
    setSelectedPresetId('');
    setIsDragging(false);
  }, []);

  // Commit UI parameters to actual parameters when dragging ends
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    setParameters(nextParameters ?? uiParameters);
    setIsDragging(false);
  }, [uiParameters]);

  // Sync UI parameters with committed parameters when they change (unless we're dragging)
  useEffect(() => {
    if (!isDragging) {
      setUiParameters(parameters);
    }
  }, [parameters, isDragging]);

  useEffect(() => {
    if (uiParameters.automation_interp_type !== 'linear') {
      setUiParameters(prev => ({ ...prev, automation_interp_type: 'linear' }));
      setParameters(prev => ({ ...prev, automation_interp_type: 'linear' }));
    }
  }, [uiParameters.automation_interp_type]);

  useEffect(() => {
    if (!searchParams || isUrlSyncReady) {
      return;
    }

    const urlParams = new URLSearchParams(searchParams.toString());
    const state = decodeFullStateFromParams(urlParams);

    isHydratingFromUrlRef.current = true;
    setParameters(state.parameters);
    setUiParameters(state.parameters);
    setEnableCodingAutomation(state.enableCodingAutomation);
    setEnableExperimentAutomation(state.enableExperimentAutomation);
    setUseExperimentThroughputCES(state.useExperimentThroughputCES);
    setEnableSoftwareProgress(state.enableSoftwareProgress);
    setUseComputeLaborGrowthSlowdown(state.useComputeLaborGrowthSlowdown);
    setUseVariableHorizonDifficulty(state.useVariableHorizonDifficulty);

    // Allow checkbox effects to run after hydration
    setTimeout(() => {
      isHydratingFromUrlRef.current = false;
    }, 0);

    setIsUrlSyncReady(true);
  }, [searchParams, isUrlSyncReady]);

  useEffect(() => {
    if (!isUrlSyncReady) {
      return;
    }

    const nextParams = encodeFullStateToParams({
      parameters,
      enableCodingAutomation,
      enableExperimentAutomation,
      useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown,
      useVariableHorizonDifficulty,
    });

    const nextSearch = nextParams.toString();
    const currentSearch = searchParams?.toString() ?? '';

    // Only update if changed
    if (nextSearch !== currentSearch) {
      router.replace(nextSearch ? `${pathname}?${nextSearch}` : pathname, { scroll: false });
    }
  }, [parameters, enableCodingAutomation, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, useVariableHorizonDifficulty, isUrlSyncReady, router, pathname, searchParams]);

  // Load full parameter configuration on mount
  useEffect(() => {
    let isCancelled = false;

    const loadParameterConfig = async () => {
      try {
        const response = await fetch('/api/parameter-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCancelled) {
          // Unwrap the config from the {success, config} wrapper
          setAllParameters(data.config || data);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load parameter configuration:', error);
        }
      } finally {
        if (!isCancelled) {
          setIsParameterConfigReady(true);
        }
      }
    };

    loadParameterConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // API now returns values already in log form (OOMs), so no conversion needed
  const toEffectiveComputeOOM = useCallback((value?: number | null): number | null => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return null;
    }
    // Values are already in OOM form from the API, just pass through with clamping
    return value
  }, []);

  const normalizeTrajectoryPoint = useCallback((point: ChartDataPoint): ChartDataPoint => {
    const normalized: ChartDataPoint = {
      ...point,
      // Values already in OOM form from API
      effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
    };

    Object.keys(point).forEach(key => {
      if (key.startsWith('effective_compute_trajectory_')) {
        const original = point[key] as number | undefined | null;
        // Values already in OOM form from API
        normalized[key] = toEffectiveComputeOOM(original);
      }
    });

    return normalized;
  }, [toEffectiveComputeOOM]);

  const isVisibleYear = useCallback((year: number | null | undefined): boolean => {
    if (year === null || year === undefined) {
      return false;
    }
    return year >= VISIBLE_CHART_START_YEAR;
  }, []);

  const fetchChartData = useCallback(
    async (params: ParametersType, signal: AbortSignal, requestId: number) => {
      try {
        const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
        const response = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            parameters: apiParameters,
            time_range: [2012, SIMULATION_END_YEAR],
            initial_progress: 0.0
          }),
          signal
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (signal.aborted) {
          return false;
        }

        if (!data.success) {
          throw new Error(data.error || 'Failed to compute model');
        }

        if (requestId !== latestChartRequestIdRef.current) {
          return false;
        }

        // Store horizon params if available
        if (data.horizon_params) {
          setHorizonParams(data.horizon_params);
        }

        // Store summary data including beta_software
        if (data.summary) {
          setSummary(data.summary);
        }

        setMilestones(
          data.milestones && typeof data.milestones === 'object'
            ? (data.milestones as MilestoneMap)
            : null
        );

        if (data.time_series && data.time_series.length > 0) {
          const formattedData: ChartDataPoint[] = data.time_series
            .filter((point: TimeSeriesPoint) => isVisibleYear(point.year))
            .sort((a: TimeSeriesPoint, b: TimeSeriesPoint) => a.year - b.year)
            .map((point: TimeSeriesPoint) => ({
              year: point.year,
              horizonLength: point.horizonLength,
              horizonFormatted: formatWorkTimeDuration(point.horizonLength),
              effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
              automationFraction: point.automationFraction,
              trainingCompute: point.trainingCompute ? toEffectiveComputeOOM(point.trainingCompute) : null,
              experimentCapacity: point.experimentCapacity,
              aiResearchTaste: point.aiResearchTaste,
              aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
              aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
              serialCodingLaborMultiplier: point.serialCodingLaborMultiplier,
              humanLabor: point.humanLabor ?? null,
              inferenceCompute: point.inferenceCompute ?? null,
              experimentCompute: point.experimentCompute ?? null,
              researchEffort: point.researchEffort ?? null,
              researchStock: point.researchStock ?? null,
              softwareProgressRate: point.softwareProgressRate ?? null,
              softwareEfficiency: point.softwareEfficiency ?? null,
              aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
              aiCodingLaborMultRefPresentDay: point.aiCodingLaborMultRefPresentDay ?? null,
            }));

          setChartData(formattedData);
          startChartAnimation(formattedData);
        }

        return true;
      } catch (error) {
        setMilestones(null);
        if (signal.aborted) {
          return false;
        }
        console.error('Error updating chart:', error);
        return false;
      }
    },
    [isVisibleYear, toEffectiveComputeOOM, startChartAnimation]
  );

  // Update the SC horizon line immediately while the slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters.ac_time_horizon_minutes);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters.ac_time_horizon_minutes]);

  // Load dynamic Monte Carlo trajectories with streaming based on current slider values
  const loadDynamicMonteCarloTrajectories = useCallback(async (allParams: typeof parameters) => {
    // Cancel any existing request
    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }

    // Create new AbortController for this request
    const abortController = new AbortController();
    currentMonteCarloRequestRef.current = abortController;

    const trajectoryMap: { [year: number]: ChartDataPoint } = {};

    try {
      const response = await fetch('/api/monte-carlo-dynamic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sliderParams: allParams,
          timeRange: [2012, SIMULATION_END_YEAR]
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let buffer = '';
      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last potentially incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const message = JSON.parse(line);

            if (message.type === 'trajectory' && message.data && Array.isArray(message.data)) {
              // Dynamic endpoint streams individual trajectory arrays - merge by year
              message.data.forEach((point: ChartDataPoint) => {
                if (!isVisibleYear(point.year)) {
                  return;
                }

                const normalizedPoint = normalizeTrajectoryPoint(point);
                const year = normalizedPoint.year;
                if (!trajectoryMap[year]) {
                  trajectoryMap[year] = normalizedPoint;
                }

                Object.keys(normalizedPoint).forEach(key => {
                  if (key.startsWith('trajectory_') || key.startsWith('effective_compute_trajectory_')) {
                    trajectoryMap[year][key] = normalizedPoint[key];
                  }
                });
              });

              // Update trajectories progressively as each one completes
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'complete') {
              // Final update when all trajectories are complete
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'trajectory_error') {
              console.warn(`Failed to generate trajectory ${message.index}:`, message.error);
            } else if (message.type === 'error') {
              throw new Error(message.error);
            }
          } catch (parseError) {
            console.warn('Failed to parse streaming line:', line, parseError);
          }
        }
      }

    } catch (error: unknown) {
      // Ignore aborted requests - they were cancelled intentionally
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      console.error('Failed to load dynamic Monte Carlo trajectories:', error);
      setTrajectoryData([]);
    } finally {
      // Clear the request reference when done (whether success or error)
      if (currentMonteCarloRequestRef.current === abortController) {
        currentMonteCarloRequestRef.current = null;
      }
    }
  }, [normalizeTrajectoryPoint, isVisibleYear]);

  const scheduleMonteCarloUpdate = useCallback((allParams: typeof parameters, skipDelay: boolean = false) => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
    }

    setTrajectoryData([]);
    if (skipDelay) {
      loadDynamicMonteCarloTrajectories(allParams);
      return;
    }
    monteCarloTimeoutRef.current = setTimeout(() => {
      monteCarloTimeoutRef.current = null;
      loadDynamicMonteCarloTrajectories(allParams);
    }, 500);
  }, [loadDynamicMonteCarloTrajectories]);

  const cancelMonteCarloUpdate = useCallback(() => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
      monteCarloTimeoutRef.current = null;
    }

    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!isParameterConfigReady) {
      return;
    }

    const paramsSnapshot = { ...parameters };
    const controller = new AbortController();

    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
    }
    chartRequestControllerRef.current = controller;

    const requestId = latestChartRequestIdRef.current + 1;
    latestChartRequestIdRef.current = requestId;

    let isActive = true;
    setMainLoading(true);
    // scheduleMonteCarloUpdate(paramsSnapshot, true);

    (async () => {
      const didUpdate = await fetchChartData(paramsSnapshot, controller.signal, requestId);
      const isCurrentRequest = latestChartRequestIdRef.current === requestId;

      if (!isActive || controller.signal.aborted || !isCurrentRequest) {
        return;
      }

      setMainLoading(false);
      if (!didUpdate) {
        cancelMonteCarloUpdate();
      }
    })();

    return () => {
      isActive = false;
      if (chartRequestControllerRef.current === controller) {
        chartRequestControllerRef.current = null;
      }
      controller.abort();
      cancelMonteCarloUpdate();
      if (latestChartRequestIdRef.current === requestId) {
        setMainLoading(false);
      }
    };
  }, [parameters, fetchChartData, scheduleMonteCarloUpdate, cancelMonteCarloUpdate, isParameterConfigReady]);

  useEffect(() => () => {
    cancelMonteCarloUpdate();
    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
      chartRequestControllerRef.current = null;
    }
  }, [cancelMonteCarloUpdate]);

  // Sync checkbox states with parameter values
  // When checked (default): uses DEFAULT_PARAMETERS
  // When unchecked: applies nonDefaultParameterValues and locks parameters
  useCheckboxParameterSync({
    checkboxValue: enableCodingAutomation,
    checkboxKey: 'enableCodingAutomation',
    nonDefaultParameterValues: {
      optimal_ces_eta_init: 0.0000001,
      coding_automation_efficiency_slope: 0.000000000001,
      swe_multiplier_at_present_day: 1.0000001,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: enableExperimentAutomation,
    checkboxKey: 'enableExperimentAutomation',
    nonDefaultParameterValues: {
      median_to_top_taste_multiplier: 1.000000000001,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: useExperimentThroughputCES,
    checkboxKey: 'useExperimentThroughputCES',
    nonDefaultParameterValues: {
      direct_input_exp_cap_ces_params: true,
      inf_labor_asymptote: 1.26697e30,
      inf_compute_asymptote: 5.19636e44,
      inv_compute_anchor_exp_cap: 2.8,
      parallel_penalty: 0.5,
      rho_experiment_capacity: 0,
      alpha_experiment_capacity: 0.699,
      experiment_compute_exponent: 0.640,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: enableSoftwareProgress,
    checkboxKey: 'enableSoftwareProgress',
    nonDefaultParameterValues: {
      software_progress_rate_at_reference_year: 0,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: useVariableHorizonDifficulty,
    checkboxKey: 'useVariableHorizonDifficulty',
    nonDefaultParameterValues: {
      doubling_difficulty_growth_factor: 1,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });



  const resolvedChartData = animatedChartData.length > 0 ? animatedChartData : chartData;

  // Merge chartData with trajectoryData for rendering (computed, not stored)
  const renderData = useMemo(() => {
    if (!Array.isArray(trajectoryData) || trajectoryData.length === 0) {
      return resolvedChartData;
    }

    const trajectoryLookup = new Map<string, ChartDataPoint>();
    trajectoryData.forEach(tp => {
      const key = Number(tp.year).toFixed(6);
      trajectoryLookup.set(key, tp);
    });

    const merged = resolvedChartData.map(point => {
      const lookupKey = Number(point.year).toFixed(6);
      const trajectoryPoint = trajectoryLookup.get(lookupKey);
      if (!trajectoryPoint) {
        return point;
      }

      const trajectoryFields: Record<string, string | number | null | undefined> = {};
      Object.keys(trajectoryPoint).forEach(key => {
        const rawValue = trajectoryPoint[key as keyof ChartDataPoint];

        if (key.startsWith('trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric) || numeric <= 0) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric
          }
        } else if (key.startsWith('effective_compute_trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric)) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric;
          }
        }
      });
      return { ...point, ...trajectoryFields };
    });

    return merged;
  }, [resolvedChartData, trajectoryData]);

  // Calculate the year when coding automation horizon is reached
  const codingAutomationHorizonYear = useMemo(() => {
    // Find the first year where horizonLength >= scHorizonMinutes
    const point = resolvedChartData.find(p =>
      p.horizonLength != null &&
      typeof p.horizonLength === 'number' &&
      p.horizonLength >= scHorizonMinutes
    );
    return point?.year ?? null;
  }, [resolvedChartData, scHorizonMinutes]);

  const asiMilestoneYear = useMemo(() => {
    if (!milestones) {
      return null;
    }
    const raw = milestones['ASI']?.time;
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
  }, [milestones]);

  // Dynamic display end year: show at least to 2030, extend to show automation achievement, max 2045
  const displayEndYear = useMemo(() => {
    const MIN_END_YEAR = 2030;
    const MAX_END_YEAR = SIMULATION_END_YEAR; // 2045

    const targetYear = asiMilestoneYear ?? codingAutomationHorizonYear;
    if (targetYear == null) {
      return MAX_END_YEAR; // Show full simulation if no milestones
    }

    return Math.min(MAX_END_YEAR, Math.max(MIN_END_YEAR, targetYear) + 0.5);
  }, [asiMilestoneYear, codingAutomationHorizonYear]);

  const automationReferenceLine = useMemo(() => {
    if (codingAutomationHorizonYear == null) {
      return undefined;
    }

    return {
      x: codingAutomationHorizonYear,
      stroke: CODING_AUTOMATION_MARKER_COLOR,
      strokeWidth: 1,
      strokeOpacity: 0.8,
    };
  }, [codingAutomationHorizonYear]);

  const isInitialMilestoneLoading = mainLoading && resolvedChartData.length === 0;

  // Small chart configurations and paging (snap per page)
  type SmallChartDef = {
    key: string;
    title: string;
    tooltip: (p: DataPoint) => React.ReactNode;
    yScale?: 'linear' | 'log';
    domain?: [number, number];
    yFormatter?: (v: number) => string | React.ReactNode;
    // Explicit tag to control layout rows irrespective of yScale defaulting
    scaleType: 'linear' | 'log';
    logSuffix?: string;
  };

  // Tooltip for Horizon Length chart
  const HorizonTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const horizonLength = 'horizonLength' in point ? point.horizonLength : null;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {horizonLength != null && typeof horizonLength === 'number' && !isNaN(horizonLength) && (
          <span style={tooltipValueStyle}>{formatWorkTimeDurationDetailed(horizonLength as number)}</span>
        )}
      </div>
    );
  }, []);

  // Tooltip for AI R&D Progress Multiplier chart
  const AIProgressMultiplierTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const value = 'aiSwProgressMultRefPresentDay' in point ? point.aiSwProgressMultRefPresentDay : null;
    const milestoneLabel = typeof (point as Record<string, unknown>)['milestoneLabel'] === 'string'
      ? (point as Record<string, unknown>)['milestoneLabel'] as string
      : undefined;
    const explanation = milestoneLabel ? MILESTONE_EXPLANATIONS[milestoneLabel] : undefined;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {value != null && typeof value === 'number' && !isNaN(value) && value > 0 && (
          <span style={tooltipValueStyle}>{formatAsPowerOfTenNode(value as number, { suffix: ' x' })}</span>
        )}
        {explanation && (
          <>
            <div className="font-semibold">{milestoneLabel}</div>
            <span style={{ color: 'var(--vivid-foreground)' }}>{explanation}</span>
          </>
        )}
      </div>
    );
  }, []);

  type PrimaryChartConfig = {
    id: string;
    title: string;
    render: () => React.ReactNode;
  };

  const primaryCharts: PrimaryChartConfig[] = useMemo(
    () => [
      {
        id: 'primary-ai-software-rd',
        title: 'AI Software R&D Uplift',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <AIRnDProgressMultiplierChart
              chartData={renderData}
              tooltip={AIProgressMultiplierTooltip}
              milestones={milestones}
              displayEndYear={displayEndYear}
              verticalReferenceLine={automationReferenceLine}
            />
          </ChartLoadingOverlay>
        ),
      },
      {
        id: 'primary-coding-horizon',
        title: 'Coding Time Horizon',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <CustomHorizonChart
              chartData={renderData}
              scHorizonMinutes={scHorizonMinutes}
              tooltip={HorizonTooltip}
              formatTimeDuration={formatWorkTimeDuration}
              benchmarkData={benchmarkData}
              className="flex flex-col"
              displayEndYear={displayEndYear}
              height={CHART_LAYOUT.primary.height}
            />
          </ChartLoadingOverlay>
        ),
      },
      {
        id: 'primary-effective-compute',
        title: 'Effective & Training Compute',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <CombinedComputeChart
              chartData={renderData}
              title={enableSoftwareProgress ? 'Effective & Training Compute' : 'Effective Compute'}
              verticalReferenceLine={automationReferenceLine}
              className="flex flex-col"
              displayEndYear={displayEndYear}
              height={CHART_LAYOUT.primary.height}
              showTrainingSeries={enableSoftwareProgress}
            />
          </ChartLoadingOverlay>
        ),
      },
    ],
    [
      AIProgressMultiplierTooltip,
      HorizonTooltip,
      automationReferenceLine,
      benchmarkData,
      displayEndYear,
      enableSoftwareProgress,
      formatWorkTimeDuration,
      mainLoading,
      milestones,
      renderData,
      scHorizonMinutes,
    ],
  );

  const primaryChartItems = useMemo(
    () => primaryCharts.map(({ id, title }) => ({ id, title })),
    [primaryCharts],
  );

  const visiblePrimaryCharts = useMemo(
    () => primaryCharts.filter((chart) => !hiddenCharts.has(chart.id)),
    [primaryCharts, hiddenCharts],
  );
  const visiblePrimaryCount = visiblePrimaryCharts.length;
  const primaryGridWidth = visiblePrimaryCount > 0
    ? visiblePrimaryCount * CHART_LAYOUT.primary.width
    : 0;


  const smallCharts: SmallChartDef[] = useMemo(() => ([
    {
      key: 'automationFraction',
      title: 'Coding Automation Fraction',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="automationFraction" formatter={(v) => `${(v * 100).toFixed(1)}%`} />,
      yFormatter: (v) => `${(v * 100).toFixed(1)}%`,
      scaleType: 'linear'
    },
    {
      key: 'aiCodingLaborMultiplier',
      title: 'AI Parallel Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiCodingLaborMultiplier" formatter={(v) => formatAsPowerOfTenNode(v, { suffix: ' x' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: ' x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' x' })
    },
    {
      key: 'aiResearchTaste',
      title: 'AI Experiment Selection Skill',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiResearchTaste" formatter={(v) => formatAsPowerOfTenNode(v)} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'researchEffort',
      title: 'Software Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchEffort" formatter={(v) => formatAsPowerOfTenNode(v)} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'researchStock',
      title: 'Cumulative Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchStock" formatter={(v) => formatAsPowerOfTenNode(v)} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'softwareProgressRate',
      title: 'Software Efficiency Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareProgressRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'softwareEfficiency',
      title: 'Software Efficiency',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareEfficiency" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'experimentCapacity',
      title: 'Experiment Throughput',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCapacity" formatter={formatTo3SigFigs} />,
      yScale: 'log',
      scaleType: 'log'
    },
    {
      key: 'inferenceCompute',
      title: 'Inference Compute for Coding Automation',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="inferenceCompute" formatter={(v) => formatAsPowerOfTenNode(v, { suffix: ' FLOP' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'experimentCompute',
      title: 'Experiment Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCompute" formatter={(v) => formatAsPowerOfTenNode(v, { suffix: ' FLOP' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'humanLabor',
      title: 'Human Coding Labor',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="humanLabor" formatter={(v) => formatAsPowerOfTenNode(v)} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
  ]), []);

  const metricChartItems = useMemo(
    () => smallCharts.map((chart) => ({ id: `metric-${chart.key}`, title: chart.title })),
    [smallCharts],
  );

  const visibleSmallCharts = useMemo(
    () => smallCharts.filter((chart) => !hiddenCharts.has(`metric-${chart.key}`)),
    [smallCharts, hiddenCharts],
  );

  type ColumnDef = { top?: SmallChartDef; middle?: SmallChartDef; bottom?: SmallChartDef };
  const columns: ColumnDef[] = useMemo(() => {
    const logs = visibleSmallCharts.filter(c => c.scaleType === 'log');
    const linears = visibleSmallCharts.filter(c => c.scaleType !== 'log');
    const numCols = Math.max(Math.ceil(logs.length / 2), linears.length);
    return Array.from({ length: numCols }, (_, i) => ({
      top: logs[2 * i],
      middle: logs[2 * i + 1],
      bottom: linears[i],
    })).filter(col => col.top || col.middle || col.bottom);
  }, [visibleSmallCharts]);
  const hasLogCharts = useMemo(() => columns.some(col => col.top || col.middle), [columns]);
  const hasLinearCharts = useMemo(() => columns.some(col => col.bottom), [columns]);
  const hasMetricCharts = hasLogCharts || hasLinearCharts;

  const SOFTWARE_PROGRESS_CHART_ID = 'primary-ai-software-rd';

  useEffect(() => {
    setHiddenCharts((prev) => {
      const next = new Set(prev);
      const shouldHide = !enableSoftwareProgress;
      const currentlyHidden = next.has(SOFTWARE_PROGRESS_CHART_ID);
      if (shouldHide && !currentlyHidden) {
        next.add(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      if (!shouldHide && currentlyHidden) {
        next.delete(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      return prev;
    });
  }, [enableSoftwareProgress]);

  return (
    <ParameterHoverProvider>
      <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
        <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
          <div
            ref={scrollContainerRef}
            className="flex min-h-0 flex-col overflow-y-auto pb-10"
          >
            <HeaderContent variant="inline" className="pt-6 pb-4 pl-8" />
            {resolvedChartData.length === 0 && !mainLoading && (
              <div className="text-center py-8 text-red-500 text-lg">No chart data available</div>
            )}

            <main className="w-full px-6">
              {/* Charts Container */}
              <div className="flex flex-col flex-1 min-h-0 pl-2">
                <ChartSyncProvider>
                  <div className="relative w-full">
                    <div className="overflow-x-auto overflow-y-hidden scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
                    <div
                      className="grid items-start gap-6"
                      style={{
                        gridTemplateColumns: `${CHART_LAYOUT.keyStats.width}px ${visiblePrimaryCount > 0 ? `${primaryGridWidth}px` : '0px'} ${hasMetricCharts ? 'minmax(0,1fr)' : '0px'}`,
                      }}
                    >
                    <div className="row-span-2">
                      <div>
                        <KeyMilestonePanel
                          automationYear={codingAutomationHorizonYear}
                          asiYear={asiMilestoneYear}
                          isLoading={isInitialMilestoneLoading}
                          primaryCharts={primaryChartItems}
                          metricCharts={metricChartItems}
                          hiddenCharts={hiddenCharts}
                          onToggleChart={toggleChartVisibility}
                        />
                      </div>
                    </div>
                    {visiblePrimaryCount > 0 && (
                      <div
                        className="col-start-2 row-span-2 grid gap-4"
                        style={{
                          gridTemplateColumns: `repeat(${visiblePrimaryCount}, minmax(0, 1fr))`,
                        }}
                      >
                        {visiblePrimaryCharts.map((chart) => (
                          <div key={chart.id} className="flex flex-col">
                            {chart.render()}
                          </div>
                        ))}
                      </div>
                    )}
                    {hasMetricCharts && (
                    <div className="col-start-3 row-span-2 min-h-0" style={{ height: CHART_LAYOUT.primary.height }}>
                      <div className="h-full">
                        <div className="flex h-full min-w-max flex-col px-1 pr-10">
                          <div className="flex items-center gap-2 pb-2">
                            <span className="font-mono small-caps text-[12px] font-semibold">Log Scale</span>
                            <div className="border-t border-gray-500/30 flex-1" />
                          </div>
                          <div className="flex flex-row gap-4 flex-[2] min-h-0">
                            {hasLogCharts ? (
                              columns.map((col, columnIndex) => (
                                <div
                                  key={`log-col-${columnIndex}`}
                                  className="pr-1 h-full snap-start"
                                  style={{ minWidth: CHART_LAYOUT.metric.columnMinWidth }}
                                >
                                  <div className="grid h-full min-h-0 grid-rows-[repeat(2,minmax(0,1fr))] gap-6">
                                    {[col.top, col.middle].map((cfg, rowIndex) => (
                                      cfg ? (
                                        <ChartLoadingOverlay key={`${cfg.key}-${columnIndex}-${rowIndex}`} isLoading={mainLoading} className="h-full">
                                          <CustomMetricChart
                                            chartData={renderData}
                                            tooltip={cfg.tooltip}
                                            title={cfg.title}
                                            dataKey={cfg.key}
                                            yFormatter={cfg.yFormatter || ((v) => formatTo3SigFigs(v))}
                                            yScale={cfg.yScale || 'linear'}
                                            domainOverride={cfg.domain}
                                            verticalReferenceLine={automationReferenceLine}
                                            className="h-full"
                                            displayEndYear={displayEndYear}
                                            logSuffix={cfg.logSuffix}
                                          />
                                        </ChartLoadingOverlay>
                                      ) : (
                                        <div
                                          key={`log-placeholder-${columnIndex}-${rowIndex}`}
                                          className="h-full rounded-lg border border-dashed border-slate-200/60"
                                        />
                                      )
                                    ))}
                                  </div>
                                </div>
                              ))
                            ) : (
                              <div className="flex min-h-[120px] flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                All log-scale charts are hidden. Toggle any chart on the left to show it again.
                              </div>
                            )}
                          </div>
                          <div className="flex items-center gap-2 py-2">
                            <span className="font-mono small-caps text-[12px] font-semibold">Linear Scale</span>
                            <div className="border-t border-gray-500/30 flex-1" />
                          </div>
                          <div className="flex flex-row gap-4 flex-1 min-h-0 mr-16">
                            {hasLinearCharts ? (
                              columns.map((col, columnIndex) => (
                                <div
                                  key={`lin-col-${columnIndex}`}
                                  className="pr-1 h-full snap-start"
                                  style={{ minWidth: CHART_LAYOUT.metric.columnMinWidth }}
                                >
                                  <div className="h-full">
                                    {col.bottom ? (
                                      <ChartLoadingOverlay key={`${col.bottom.key}-${columnIndex}-linear`} isLoading={mainLoading} className="h-full">
                                        <CustomMetricChart
                                          chartData={renderData}
                                          tooltip={col.bottom.tooltip}
                                          title={col.bottom.title}
                                          dataKey={col.bottom.key}
                                          yFormatter={col.bottom.yFormatter || ((v) => formatTo3SigFigs(v))}
                                          yScale={col.bottom.yScale || 'linear'}
                                          domainOverride={col.bottom.domain}
                                          verticalReferenceLine={automationReferenceLine}
                                          className="h-full"
                                          displayEndYear={displayEndYear}
                                          logSuffix={col.bottom.logSuffix}
                                        />
                                      </ChartLoadingOverlay>
                                      ) : (
                                        <div className="h-full rounded-lg border border-dashed border-slate-200/60" />
                                      )}
                                  </div>
                                </div>
                              ))
                            ) : (
                              <div className="flex min-h-[80px] flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                All linear-scale charts are hidden. Toggle any chart on the left to show it again.
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    )}
                      </div>
                    </div>
                    <div
                      className="pointer-events-none absolute inset-y-0 right-0 w-32"
                      style={{
                        background:
                          'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.9) 50%, var(--vivid-background, #ffffff) 100%)',
                      }}
                    />
                </div>
                </ChartSyncProvider>
              </div>
              <div className="flex flex-row items-center gap-3 space-around small-caps font-system-mono">
                {/* <div className="flex-1 border-t border-gray-500/30" /> */}
                <span className="px-2 text-xs font-semibold tracking-[0.2em] uppercase">Key Parameters</span>
                <div className="flex-1 border-t border-gray-500/30" />
                <button
                  type="button"
                  onClick={handleResetKeyParameters}
                  className="rounded-sm border border-[#195227] bg-[#2A623D] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.3em] text-white shadow-[0_1px_2px_rgba(32,83,53,0.35)] transition hover:bg-[#245632] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2A623D]"
                >
                  Reset
                </button>
              </div>
              <section className="w-full px-2 mt-4">
                <div className="grid grid-cols-5 gap-10">
                  {/* AI Research Taste Slope Slider */}
                  <div className="relative flex w-full flex-col justify-between">
                    <div className="flex items-start justify-between mb-1">
                      <label className="flex w-full text-sm font-medium text-foreground font-system-mono">
                        <WithChartTooltip
                          explanation="Each time effective compute increases by as much as it does in 2025, by how many standard deviations (SDs) does AI experiment selection skill increase (median to best AGI company researcher requires 3 SDs)? This parameter is called AI experiment selection slope in the writeup + parameter export/import."
                          fullWidth
                        >
                          How quickly do AIs improve at experiment selection?
                        </WithChartTooltip>
                      </label>
                    </div>
                    <ParameterSlider
                      paramName="ai_research_taste_slope"
                      label=""
                      step={0.1}
                      fallbackMin={0.1}
                      fallbackMax={10.0}
                      decimalPlaces={1}
                      value={uiParameters.ai_research_taste_slope}
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      useLogScale={true}
                    />
                  </div>
                  {/* Median-to-Top Taste Multiplier Slider */}
                  <div className="relative flex w-full flex-col justify-between">
                    <div className="flex items-start justify-between mb-1">
                      <label className="block w-full text-sm font-medium text-foreground font-system-mono">
                        <WithChartTooltip
                          explanation="Ratio between the experiment selection skill of the top AGI project researcher and the median project researcher. This parameter is called median-to-top experiment selection multiplier in the writeup + parameter export/import."
                          fullWidth
                        >
                          Ratio between Median and Top Human Researchers&apos; Experiment Selection Skill
                        </WithChartTooltip>
                      </label>
                    </div>
                    <ParameterSlider
                      paramName="median_to_top_taste_multiplier"
                      label=""
                      step={0.1}
                      fallbackMin={1.1}
                      fallbackMax={20.0}
                      decimalPlaces={2}
                      value={uiParameters.median_to_top_taste_multiplier}
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      disabled={lockedParameters.has('median_to_top_taste_multiplier')}
                      useLogScale={true}
                    />
                  </div>
                  {/* Present Doubling Time Slider */}
                  <div className="relative flex w-full flex-col justify-between">
                    <div className="flex items-center justify-between mb-1">
                      <label className="block w-full text-sm font-medium text-foreground font-system-mono">
                        <WithChartTooltip
                          explanation="How long it takes the coding time horizon to double, beginning at present day but extended as a long-term trend."
                          fullWidth
                        >
                          Present Time Horizon Doubling Time
                        </WithChartTooltip>
                      </label>
                    </div>
                    <ParameterSlider
                      paramName="present_doubling_time"
                      label="" // Label already rendered above
                      fallbackMin={0.01}
                      fallbackMax={2.0}
                      step={0.01}
                      decimalPlaces={2}
                      customFormatValue={(years) => formatTimeDuration(yearsToMinutes(years))}
                      value={uiParameters.present_doubling_time}
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      onMouseEnter={() => handlePopoverOpen('doubling_time')}
                      onMouseLeave={handlePopoverClose}
                      onFocus={() => handlePopoverOpen('doubling_time')}
                      onBlur={handlePopoverClose}
                      popoverConfig={CHART_CONFIGS.doubling_time}
                      popoverVisible={activePopover === 'doubling_time'}
                      onPopoverClose={handlePopoverClose}
                      horizonParams={horizonParams}
                      useLogScale={true}
                    />
                  </div>

                  {/* SC Time Horizon Slider */}
                  <div className="relative flex w-full flex-col justify-between">
                    <div className="flex items-center justify-between mb-1">
                      <label className="block w-full text-sm font-medium text-foreground font-system-mono">
                        <WithChartTooltip
                          explanation="If an AI can do 80% of tasks that take a skilled human this long, then an AI can do every type of coding work as well as the average company engineer."
                          fullWidth
                        >
                          Automated Coder Time Horizon Requirement
                        </WithChartTooltip>
                      </label>
                    </div>
                    <ParameterSlider
                      paramName="ac_time_horizon_minutes"
                      label="" // Label already rendered above
                      customMin={scHorizonLogBounds.min}
                      customMax={scHorizonLogBounds.max}
                      step={0.1}
                      customFormatValue={formatSCHorizon}
                      value={uiParameters.ac_time_horizon_minutes}
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      onMouseEnter={() => handlePopoverOpen('sc_horizon')}
                      onMouseLeave={handlePopoverClose}
                      onFocus={() => handlePopoverOpen('sc_horizon')}
                      onBlur={handlePopoverClose}
                      popoverConfig={CHART_CONFIGS.sc_horizon}
                      popoverVisible={activePopover === 'sc_horizon'}
                      onPopoverClose={handlePopoverClose}
                      horizonParams={horizonParams}
                      useLogScale={true}
                    />
                  </div>

                  {/* Doubling Difficulty Growth Factor Slider */}
                  <div className="relative flex w-full flex-col justify-between">
                    <div className="flex items-center justify-between mb-1">
                      <label className="block w-full text-sm font-medium text-foreground font-system-mono">
                        <WithChartTooltip
                          explanation='For each successive time horizon doubling, by what factor does the required orders of magnitude of effective compute change? Lower values mean doublings get easier over time. This parameter is called doubling difficulty growth factor in the writeup + parameter export/import.'
                          fullWidth
                          tooltipPlacement="left"
                        >
                          How much easier/harder each time horizon doubling gets (higher is harder)
                        </WithChartTooltip>
                      </label>
                    </div>
                    <ParameterSlider
                      paramName="doubling_difficulty_growth_factor"
                      label="" // Label already rendered above
                      fallbackMin={0.5}
                      fallbackMax={1.5}
                      step={0.01}
                      decimalPlaces={3}
                      value={uiParameters.doubling_difficulty_growth_factor}
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      onMouseEnter={() => handlePopoverOpen('difficulty_growth_rate')}
                      onMouseLeave={handlePopoverClose}
                      onFocus={() => handlePopoverOpen('difficulty_growth_rate')}
                      onBlur={handlePopoverClose}
                      popoverConfig={CHART_CONFIGS.difficulty_growth_rate}
                      popoverVisible={activePopover === 'difficulty_growth_rate'}
                      onPopoverClose={handlePopoverClose}
                      horizonParams={horizonParams}
                      disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                    />
                  </div>
                </div>
              </section>

              {/* Model Simplification Controls */}
              <div className="flex flex-row gap-4 space-around small-caps font-system-mono items-center mt-10">
                {/* <div className="flex-1 border-t border-gray-500/30" /> */}
                <span className="px-2 text-xs font-semibold tracking-[0.2em] uppercase">Model Features</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <section className="w-full px-2 mt-6 mb-4">
                <div className="flex justify-between gap-8 font-system-mono">
                  <ModelFeatureCheckbox
                    id="enable-coding-automation"
                    label="Coding Automation"
                    checked={enableCodingAutomation}
                    onChange={setEnableCodingAutomation}
                    hoverText="When disabled: Sets automation parameters to near-zero, effectively removing AI coding automation from the model."
                  />

                  <ModelFeatureCheckbox
                    id="enable-experiment-automation"
                    label="Experiment Selection Automation"
                    checked={enableExperimentAutomation}
                    onChange={setEnableExperimentAutomation}
                    hoverText="When disabled: Sets the median-to-top taste multiplier to near-one, removing the advantage of better experiment selection skill."
                  />

                  <ModelFeatureCheckbox
                    id="enable-software-progress"
                    label="Software Progress"
                    checked={enableSoftwareProgress}
                    onChange={setEnableSoftwareProgress}
                    hoverText="When disabled: Sets software efficiency growth rate to zero, meaining AI progress only comes from traning compute."
                  />

                  <ModelFeatureCheckbox
                    id="use-variable-horizon-difficulty"
                    label="Variable Horizon Doubling Difficulty"
                    checked={useVariableHorizonDifficulty}
                    onChange={setUseVariableHorizonDifficulty}
                    hoverText="When disabled: Locks doubling difficulty growth factor to 1, making successive time horizon doublings require the same change in effective compute (matching an exponential fit to the past data)."
                  />

                  <ModelFeatureCheckbox
                    id="use-experiment-throughput-ces"
                    label="Experiment Throughput CES"
                    checked={useExperimentThroughputCES}
                    onChange={setUseExperimentThroughputCES}
                    hoverText="When enabled: Uses CES production function for experiment throughput instead of Cobb-Douglas, meaning that taking only one of experiment compute or coding labor to infinity bottlenecks at a fixed limit."
                  />

                  {/* <ModelFeatureCheckbox
                    id="use-compute-labor-growth-slowdown"
                    label="Compute+Labor Growth Slowdown [disabled]"
                    checked={useComputeLaborGrowthSlowdown}
                    onChange={setUseComputeLaborGrowthSlowdown}
                    hoverText="When disabled: Assumes compute scaling has a constant growth rate matching 2025, including in the past, rather than using historical data."
                    // TODO: re-enable after the functinonality is implemented
                    disabled
                  /> */}
                </div>
              </section>
              <div className="flex flex-row items-center gap-3 space-around small-caps font-system-mono mt-10">
                {/* <div className="flex-1 border-t border-gray-500/30" /> */}
                <span className="px-2 text-xs font-semibold tracking-[0.2em] uppercase" id="model-explanation">Model Explanation</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <ModelArchitecture />
              <div className="max-w-7xl mx-auto">
              </div>
              <div className="prose max-w-3xl mx-auto mt-10">
                {/* {IntroductionMarkdown} */}
              </div>
            </main>

            {/* <ModelDiagram className="w-full hidden aspect-[2648/476] min-h-[200px] md:block" />
                <ModelDiagramTopDown className="w-full block aspect-[1267/1208] min-h-[500px] md:hidden" /> */}

          </div>
        </div>

        {/* Advanced Parameters Column */}
        <div className="col-start-2 row-start-1 flex h-full overflow-visible shadow-[0_0_1em_0_rgba(0,0,0,0.5)] z-10">
          <div
            className={`flex h-full items-stretch transition-[width] duration-300 ease-in-out flex-row-reverse ${isAdvancedPanelOpen ? 'w-[22rem]' : 'w-12'
              }`}
          >
            {isAdvancedPanelOpen && (
              <aside
                id="advanced-panel"
                className="flex-1 bg-white border-l border-gray-200 flex flex-col"
              >
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                  <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.3em] font-system-mono">
                    Advanced Parameters
                  </h2>
                  <button
                    type="button"
                    onClick={() => setIsAdvancedPanelOpen(false)}
                    className="text-xs font-medium text-gray-500 hover:text-gray-700"
                  >
                    Close
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto px-4 py-4">
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-2 sm:gap-3">
                        <button
                          type="button"
                          onClick={handleExportParameters}
                          className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.25)] hover:bg-[#245632] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                        >
                          Export Settings
                        </button>
                        <button
                          type="button"
                          onClick={handleImportButtonClick}
                          className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.1)] hover:bg-[#F4F8F5] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                        >
                          Import Settings
                        </button>
                        <input
                          ref={importInputRef}
                          type="file"
                          accept="application/json"
                          className="hidden"
                          onChange={handleImportParameters}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="block text-xs font-semibold uppercase tracking-[0.18em] text-primary font-system-mono">
                          Parameter Presets
                        </label>
                        <div className="relative">
                          <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-[11px] uppercase tracking-[0.2em] text-gray-400 font-system-mono">
                            Preset
                          </span>
                          <select
                            value={selectedPresetId}
                            onChange={handlePresetSelect}
                            className="w-full pl-20 pr-8 py-3 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-system-mono focus:outline-none focus:ring-2 focus:ring-[#2A623D] appearance-none"
                          >
                            <option value="">Select</option>
                            {PARAMETER_PRESETS.map((preset) => (
                              <option key={preset.id} value={preset.id}>
                                {preset.label}
                              </option>
                            ))}
                          </select>
                          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400">
                            <svg
                              className="h-3 w-3"
                              viewBox="0 0 12 8"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                              aria-hidden="true"
                            >
                              <path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </span>
                        </div>
                      </div>
                    </div>
                    <AdvancedSections
                      uiParameters={uiParameters}
                      setUiParameters={setUiParameters}
                      allParameters={allParameters}
                      isDragging={isDragging}
                      setIsDragging={setIsDragging}
                      commitParameters={commitParameters}
                      scHorizonLogBounds={scHorizonLogBounds}
                      preGapHorizonBounds={preGapHorizonBounds}
                      parallelPenaltyBounds={parallelPenaltyBounds}
                      summary={summary}
                      lockedParameters={lockedParameters}
                    />
                  </div>
                </div>
              </aside>
            )}
            <button
              type="button"
              onClick={() => setIsAdvancedPanelOpen((prev) => !prev)}
              className="group relative flex h-full items-center justify-center w-12 bg-black text-white hover:bg-[#245632] transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D]"
              aria-expanded={isAdvancedPanelOpen}
              aria-controls="advanced-panel"
            >
              {!isAdvancedPanelOpen && (
                <svg
                  className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full cursor-pointer"
                  width="16"
                  height="60"
                  viewBox="0 0 16 60"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M16 0 L16 60 L4 52 Q0 50 0 30 Q0 10 4 8 L16 0"
                    className="fill-black group-hover:fill-[#245632] transition-colors"
                    stroke="none"
                  />
                  <line x1="5" y1="18" x2="5" y2="42" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
                  <line x1="9" y1="16" x2="9" y2="44" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
                </svg>
              )}
              <span className="-rotate-90 whitespace-nowrap tracking-[0.4em] text-xs font-semibold">
                {isAdvancedPanelOpen ? 'CLOSE' : 'ADVANCED PARAMETERS'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </ParameterHoverProvider>
  );
}
