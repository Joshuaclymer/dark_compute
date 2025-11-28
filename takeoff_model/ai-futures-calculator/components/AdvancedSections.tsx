import React from 'react';
import { ParametersType } from '@/constants/parameters';
import { ParameterSlider } from './ParameterSlider';
import { formatSCHorizon, formatAsPowerOfTenText, formatWorkTimeDuration } from '@/utils/formatting';
import { formatTo3SigFigs } from '@/utils/formatting';

// Type definitions
interface ModelDefaults {
    optimal_ces_eta_init?: number;
    automation_interp_type?: string;
    ai_research_taste_slope?: number;
    anchor_progress_at_strong_cognitive_horizon?: number;
    present_year?: number;
    present_progress?: number;
    progress_at_aa?: number;
}

export interface ParameterConfig {
    defaults?: ModelDefaults;
    bounds?: Record<string, [number, number]>;
    metadata?: Record<string, unknown>;
}

// Props interface for AdvancedSections component
export interface AdvancedSectionsProps {
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;
    scHorizonLogBounds: { min: number; max: number };
    preGapHorizonBounds: { min: number; max: number };
    parallelPenaltyBounds: { min: number; max: number };
    summary?: {beta_software?: number; r_software?: number; [key: string]: unknown} | null;
    lockedParameters?: Set<string>;
}

// Extract AdvancedSections component to module scope to prevent recreation on every render
export const AdvancedSections: React.FC<AdvancedSectionsProps> = ({
    uiParameters,
    setUiParameters,
    allParameters,
    isDragging,
    setIsDragging,
    commitParameters,
    scHorizonLogBounds,
    preGapHorizonBounds,
    parallelPenaltyBounds,
    summary,
    lockedParameters,
}) => {
    const handleGapModeToggle = (checked: boolean) => {
        const next = { ...uiParameters, benchmarks_and_gaps_mode: checked };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleDirectInputToggle = (checked: boolean) => {
        const next = { ...uiParameters, direct_input_exp_cap_ces_params: checked };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleTasteScheduleChange = (value: string) => {
        const next = { ...uiParameters, taste_schedule_type: value as ParametersType['taste_schedule_type'] };
        setUiParameters(next);
        commitParameters(next);
    };

    return (
    <>
    <div className="space-y-8 mt-4">
        {/* Inputs Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">
                Inputs
            </summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                <ParameterSlider
                    paramName="constant_training_compute_growth_rate"
                    label="Pre-Slowdown Training Compute Growth Rate"
                    description="Annual growth rate of training compute before the slowdown year (OOMs/year)"
                    step={0.01}
                    fallbackMin={0.0}
                    fallbackMax={2.0}
                    decimalPlaces={2}
                    customFormatValue={(value) => `${value.toFixed(2)} OOMs/year`}
                    value={uiParameters.constant_training_compute_growth_rate}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="slowdown_year"
                    label="Slowdown Year"
                    description="Year at which training compute growth rate transitions to the post-slowdown rate"
                    step={0.1}
                    fallbackMin={2020.0}
                    fallbackMax={2040.0}
                    decimalPlaces={1}
                    customFormatValue={(value) => `${value.toFixed(1)}`}
                    value={uiParameters.slowdown_year}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="post_slowdown_training_compute_growth_rate"
                    label="Post-Slowdown Training Compute Growth Rate"
                    description="Annual growth rate of training compute after the slowdown year (OOMs/year)"
                    step={0.01}
                    fallbackMin={0.0}
                    fallbackMax={2.0}
                    decimalPlaces={2}
                    customFormatValue={(value) => `${value.toFixed(2)} OOMs/year`}
                    value={uiParameters.post_slowdown_training_compute_growth_rate}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />
            </div>
            </div>
        </details>

        {/* Time Horizon & Progress Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">
                Coding Time Horizon Requirement
                <span className="text-xs text-gray-500 font-normal ml-2">
                    {uiParameters.benchmarks_and_gaps_mode ? "(Benchmarks & Gaps Mode)" : "(Standard Mode)"}
                </span>
            </summary>
            <div className="ml-4 mt-4">

            {/* Gaps Mode Toggle */}
            <div className="mb-4 p-3 bg-gray-50 rounded-lg border">
                <div className="flex items-center justify-between">
                    <div>
                        <label className="block text-sm font-medium text-foreground font-system-mono">
                            Include an Effective Compute Gap
                        </label>
                        <div className="text-xs text-gray-500 mt-1">
                            Require additional effective compute for AC to be achieved after the time horizon requirement is met.
                        </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer ml-4">
                        <input
                            type="checkbox"
                            checked={uiParameters.benchmarks_and_gaps_mode}
                            onChange={(e) => handleGapModeToggle(e.target.checked)}
                            className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                    </label>
                </div>
            </div>

            <div className="grid grid-cols-1 gap-4 mt-4">
                {uiParameters.benchmarks_and_gaps_mode ? (
                    // Gap mode: use pre-gap horizon and explicit gap years
                    <>
                        <ParameterSlider
                            paramName="saturation_horizon_minutes"
                            label="Pre-gap AC Horizon (Target)"
                            description={`Target horizon before adding the gap`}
                            customMin={preGapHorizonBounds.min}
                            customMax={preGapHorizonBounds.max}
                            customStep={Math.max(100, preGapHorizonBounds.min / 100)}
                            customFormatValue={formatWorkTimeDuration}
                            value={uiParameters.saturation_horizon_minutes}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                        />

                        <ParameterSlider
                            paramName="gap_years"
                            label="Gap Years"
                            description="A value of 1 means the mangnitude of the gap is the effective compute increase in the present year."
                            fallbackMin={0.1}
                            fallbackMax={10.0}
                            step={0.1}
                            customFormatValue={(value) => `${value.toFixed(1)} years`}
                            value={uiParameters.gap_years}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            useLogScale={true}
                        />
                    </>
                ) : (
                    <ParameterSlider
                        paramName="ac_time_horizon_minutes"
                        label="AC Time Horizon (Target)"
                        description={`Target 80% reliability time horizon for Automated Coder determination`}
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
                        useLogScale={true}
                    />
                )}
            </div>

            {/* Mode Explanation */}
            <div className={`mt-4 text-sm p-3 rounded-lg ${
                uiParameters.benchmarks_and_gaps_mode
                    ? 'text-orange-600 bg-orange-50'
                    : 'text-blue-600 bg-blue-50'
            }`}>
                {uiParameters.benchmarks_and_gaps_mode
                    ? 'Gap mode: AC is reached when the pre-gap horizon is met and the specified gap years elapse.'
                    : 'Standard mode: AC is reached when the AC horizon threshold is met.'
                }
            </div>
            </div>
        </details>

        {/* Coding Automation Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">Coding Automation</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                <ParameterSlider
                    paramName="swe_multiplier_at_present_day"
                    label="Present Day Parallel Coding Labor Multiplier"
                    description="For what value of N would an AGI company in the present be indifferent between getting Nx more programmers  and foregoing AI usage, vs. the status quo including AI usage."
                    step={0.05}
                    fallbackMin={1.0}
                    fallbackMax={10.0}
                    decimalPlaces={2}
                    value={uiParameters.swe_multiplier_at_present_day}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    disabled={lockedParameters?.has('swe_multiplier_at_present_day')}
                    useLogScale={true}
                />

                <ParameterSlider
                    paramName="coding_automation_efficiency_slope"
                    label="Coding Automation Efficiency Slope (η)"
                    description="For a given task, each time we increase effective compute by the amount crossed in the present year (on top of the initial effective compute requirement to automate a task), by how many OOMs does the 'conversion rate' of GPUs->humans improve?"
                    step={0.1}
                    fallbackMin={0.01}
                    fallbackMax={10.0}
                    decimalPlaces={2}
                    value={uiParameters.coding_automation_efficiency_slope}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    disabled={lockedParameters?.has('coding_automation_efficiency_slope')}
                    useLogScale={true}
                />

                <ParameterSlider
                    paramName="rho_coding_labor"
                    label="Coding Labor Substitutability (ρ_c)"
                    description="This controls to what extent coding is a fixed series of tasks, vs. being able to substitute automated tasks for non-automated ones (lower values means more like the former)."
                    step={0.1}
                    fallbackMin={-10}
                    fallbackMax={0}
                    decimalPlaces={2}
                    value={uiParameters.rho_coding_labor}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                {/* TODO: This really needs to be a slider with a log scale */}
                <ParameterSlider
                    paramName="max_serial_coding_labor_multiplier"
                    label="Max Serial Coding Labor Multiplier"
                    description="At the physical limits of coding capability, AI could provide productivity benefits equivalent to speeding up all human coders by this much."
                    step={1.0}
                    stepCount={30}
                    fallbackMin={1.0}
                    fallbackMax={1e12}
                    customFormatValue={formatAsPowerOfTenText}
                    value={uiParameters.max_serial_coding_labor_multiplier}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    useLogScale
                />
            </div>
            </div>
        </details>

        {/* CES Production Functions */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">
                Experiment Throughput Production
                <span className="text-xs text-gray-500 font-normal ml-2">
                    {uiParameters.direct_input_exp_cap_ces_params ? "(Direct Input Mode)" : "(Computed from Constraints)"}
                </span>
            </summary>
            <div className="ml-4 mt-4">

            {/* Direct Input CES Params Toggle */}
            <div className="mb-4 p-3 bg-gray-50 rounded-lg border">
                <div className="flex items-center justify-between">
                    <div>
                        <label className="block text-sm font-medium text-foreground font-system-mono">
                            Direct Input CES Params
                        </label>
                        <div className="text-xs text-gray-500 mt-1">
                            Use direct CES parameter input vs. computed from constraints
                        </div>
                    </div>
                    <label className={`relative inline-flex items-center ml-4 ${lockedParameters?.has('direct_input_exp_cap_ces_params') ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
                        <input
                            type="checkbox"
                            checked={uiParameters.direct_input_exp_cap_ces_params}
                            onChange={(e) => handleDirectInputToggle(e.target.checked)}
                            disabled={lockedParameters?.has('direct_input_exp_cap_ces_params')}
                            className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600 peer-disabled:opacity-50 peer-disabled:cursor-not-allowed"></div>
                    </label>
                </div>
            </div>

            {uiParameters.direct_input_exp_cap_ces_params ? (
                // Direct Input Mode: Show CES parameters directly
                <div className="space-y-4">
                    <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
                        Direct input mode: CES parameters are used as provided
                    </div>
                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="rho_experiment_capacity"
                            label="Substitutability (ρ_x)"
                            description="Controls the degree of substitutability between experiment compute and coding labor. Lower values mean more substitutable."
                            step={0.01}
                            fallbackMin={-1}
                            fallbackMax={1}
                            decimalPlaces={3}
                            value={uiParameters.rho_experiment_capacity}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('rho_experiment_capacity')}
                        />
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="alpha_experiment_capacity"
                            label="Experiment Compute Weight (α)"
                            description="Higher values mean experiment compute is more important relative to coding labor"
                            step={0.01}
                            fallbackMin={0.05}
                            fallbackMax={0.95}
                            decimalPlaces={3}
                            value={uiParameters.alpha_experiment_capacity}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('alpha_experiment_capacity')}
                        />

                        <ParameterSlider
                            paramName="experiment_compute_exponent"
                            label="Experiment Compute Discounting (ζ)"
                            description="Experiment compute is taken to the exponent ζ before being combined with coding labor."
                            step={0.01}
                            fallbackMin={0.001}
                            fallbackMax={10}
                            decimalPlaces={3}
                            value={uiParameters.experiment_compute_exponent}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('experiment_compute_exponent')}
                        />
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="coding_labor_exponent"
                            label="Coding Parallel Penalty (λ)"
                            description="Multiplying the size of your coding labor force by Nx is equivalent to speeding up coding labor by (N^𝜆)x. This is used to convert parallel to serial coding labor before entering the CES."
                            customMin={parallelPenaltyBounds.min}
                            customMax={parallelPenaltyBounds.max}
                            step={0.01}
                            decimalPlaces={3}
                            value={uiParameters.coding_labor_exponent}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('parallel_penalty') || lockedParameters?.has('coding_labor_exponent')}
                        />
                    </div>
                </div>
            ) : (
                // Asymptote Mode: Show asymptote and anchor controls
                <div className="space-y-4">
                    <div className="text-sm text-green-600 bg-green-50 p-3 rounded-lg">
                        Constraints mode: CES parameters are computed from constraints
                    </div>
                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="inf_labor_asymptote"
                            label="Infinite Coding Labor Asymptote"
                            description="By what factor faster AI software progress would go in 2024 if you immediately got unlimited coding labor."
                            step={0.1}
                            fallbackMin={1}
                            fallbackMax={100000}
                            decimalPlaces={1}
                            value={uiParameters.inf_labor_asymptote}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('inf_labor_asymptote')}
                            useLogScale={true}
                        />
                        <ParameterSlider
                            paramName="inf_compute_asymptote"
                            label="Infinite Experiment Compute Asymptote"
                            description="By what factor faster AI software progress would go in 2024 if you immediately got unlimited experiment compute."
                            step={10}
                            fallbackMin={1}
                            fallbackMax={100000}
                            decimalPlaces={0}
                            value={uiParameters.inf_compute_asymptote}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('inf_compute_asymptote')}
                            useLogScale={true}
                        />
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="inv_compute_anchor_exp_cap"
                            label="Slowdown from 10x less Experiment Compute"
                            description="By what factor slower AI software progress would go in 2024 if you immediately had 10x less experiment compute."
                            step={0.1}
                            fallbackMin={1}
                            fallbackMax={10}
                            decimalPlaces={1}
                            value={uiParameters.inv_compute_anchor_exp_cap}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('inv_compute_anchor_exp_cap')}
                            useLogScale={true}
                        />
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                        <ParameterSlider
                            paramName="coding_labor_exponent"
                            label="Coding Parallel Penalty (λ)"
                            description="Multiplying the size of your coding labor force by Nx is equivalent to speeding up coding labor by (N^𝜆)x. This is used to convert parallel to serial coding labor before entering the CES."
                            customMin={parallelPenaltyBounds.min}
                            customMax={parallelPenaltyBounds.max}
                            step={0.01}
                            decimalPlaces={3}
                            value={uiParameters.coding_labor_exponent}
                            uiParameters={uiParameters}
                            setUiParameters={setUiParameters}
                            allParameters={allParameters}
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            commitParameters={commitParameters}
                            disabled={lockedParameters?.has('parallel_penalty') || lockedParameters?.has('coding_labor_exponent')}
                        />
                    </div>
                </div>
            )}
            </div>
        </details>

        {/* AI Research Taste Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">Experiment Selection Automation</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                <ParameterSlider
                    paramName="ai_research_taste_at_coding_automation_anchor_sd"
                    label="AI Experiment Selection Skill at AC (SDs)"
                    description="When we reach SC, how good will AIs be at experiment selection relative to the median OpenBrain research scientist (0 SDs=median, ~3 SDs=best)"
                    step={0.1}
                    fallbackMin={-10}
                    fallbackMax={23}
                    decimalPlaces={1}
                    value={uiParameters.ai_research_taste_at_coding_automation_anchor_sd}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="ai_research_taste_slope"
                    label="AI Experiment Selection Slope (SDs/present-OOMs-per-year)"
                    description="For each amount of effective OOMs crossed in the present year, by how many SDs in the OpenBrain range is AIs' experiment selection increased?"
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

                <ParameterSlider
                    paramName="median_to_top_taste_multiplier"
                    label="Median to Top Experiment Seleciton Multiplier"
                    description="Ratio of the top researcher's experiment selection skill to median researcher's skill"
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
                    disabled={lockedParameters?.has('median_to_top_taste_multiplier')}
                    useLogScale={true}
                />

                <ParameterSlider
                    paramName="taste_limit"
                    label="Maximum Experiment Selection Skill"
                    description="Number of multiplicative median-to-top experiment selection gaps between the best humans and maximally capable AIs."
                    step={0.1}
                    fallbackMin={0}
                    fallbackMax={100}
                    decimalPlaces={1}
                    value={uiParameters.taste_limit}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="taste_limit_smoothing"
                    label="Experiment Selection Slowdown Factor Halfway to Algorithmic Limit"
                    description="Halfway in log space to maximum experiment selection skill, each SD of experiment selection skill translates into the near-human-range-skill-per-SD taken to this power."
                    step={0.001}
                    fallbackMin={0.001}
                    fallbackMax={0.999}
                    decimalPlaces={3}
                    value={uiParameters.taste_limit_smoothing}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />
            </div>
            </div>
        </details>

        {/* General Capabilities Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">General Capabilities</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                <ParameterSlider
                    paramName="ted_ai_m2b"
                    label="Median-to-top SDs corresponding to TED-AI"
                    description="This multiplier on median-to-top experiment selection SDs above SAR corresponds to TED-AI."
                    useLogScale={true}
                    step={0.1}
                    fallbackMin={0}
                    fallbackMax={10}
                    decimalPlaces={1}
                    value={uiParameters.ted_ai_m2b}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="asi_above_siar_vs_tedai_above_sar_difficulty"
                    label="Difficulty factor on top of SIAR to achieve ASI, relative to TED-AI on top of SAR"
                    description="This multiplier on median-to-top SDs corresponding to TED-AI, added to SIAR's 2 median-to-top SDs, corresponds to ASI."
                    useLogScale={true}
                    step={0.1}
                    fallbackMin={0}
                    fallbackMax={10}
                    decimalPlaces={1}
                    value={uiParameters.asi_above_siar_vs_tedai_above_sar_difficulty}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />
            </div>
            </div>
        </details>

        {/* Effective Compute Parameters */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">Effective Compute</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">

                <ParameterSlider
                    paramName="software_progress_rate_at_reference_year"
                    label="Software Efficiency OOMs/year in 2024"
                    description="In OOMs/year, how quickly was software efficiency growing in 2024?"
                    step={0.1}
                    fallbackMin={0.00001}
                    fallbackMax={10}
                    decimalPlaces={2}
                    value={uiParameters.software_progress_rate_at_reference_year}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    disabled={lockedParameters?.has('software_progress_rate_at_reference_year')}
                />
            </div>

            {/* Display calibrated beta_software if available */}
            {summary?.beta_software != null && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="text-sm font-medium text-gray-700 mb-1">Calibrated Parameters</div>
                    <div className="text-xs text-gray-600">
                        <span className="font-medium text-gray-700">Calibrated r:</span>{' '}
                        <span>{summary.r_software != null ? formatTo3SigFigs(summary.r_software) : '—'}</span>
                        {', '}
                        <span className="font-medium text-gray-700">β:</span>{' '}
                        <span>{formatTo3SigFigs(summary.beta_software)}</span>
                        {' OOMs of research stock / OOM of software efficiency'}
                    </div>
                </div>
            )}
            </div>
        </details>

        {/* Extra Parameters - Not varied between simulations */}
        <details className="mb-8">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">Extra (parameters we don&apos;t vary between simulations)</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                <ParameterSlider
                    paramName="present_day"
                    label="Present Day"
                    description="Used as a reference point for setting other parameters and for capability metrics."
                    step={0.1}
                    fallbackMin={2020.0}
                    fallbackMax={2030.0}
                    decimalPlaces={1}
                    value={uiParameters.present_day}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="present_horizon"
                    label="Present Horizon"
                    description="The time horizon in present day; change if you change the present time."
                    step={0.1}
                    fallbackMin={0.01}
                    fallbackMax={100}
                    decimalPlaces={1}
                    value={uiParameters.present_horizon}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />

                <ParameterSlider
                    paramName="automation_fraction_at_coding_automation_anchor"
                    label="Coding Automation Fraction at AC"
                    description="The fraction of coding tasks efficiently automatable at AC. This is constant at 1 during our simulations, only change if you understand the model."
                    step={0.01}
                    fallbackMin={0.01}
                    fallbackMax={1.0}
                    decimalPlaces={2}
                    value={uiParameters.automation_fraction_at_coding_automation_anchor}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    disabled={lockedParameters?.has('automation_fraction_at_coding_automation_anchor')}
                />

                <ParameterSlider
                    paramName="optimal_ces_eta_init"
                    label="Initial Automation Efficiency (η_init)"
                    description="The automation efficiency of a task at the point it is considered efficiently automated (via the coding automation fraction). Measured in coding FTEs per H100be."
                    step={0.001}
                    fallbackMin={1e-12}
                    fallbackMax={1e12}
                    decimalPlaces={4}
                    value={uiParameters.optimal_ces_eta_init}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    useLogScale={true}
                />

                <ParameterSlider
                    paramName="top_percentile"
                    label="Percentile of Best Human Researcher"
                    description="The percentile of the best researcher in the AGI project human range"
                    step={0.001}
                    fallbackMin={0.5}
                    fallbackMax={0.99999}
                    decimalPlaces={5}
                    customFormatValue={(value) => `${(value * 100).toFixed(3)}%`}
                    value={uiParameters.top_percentile}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                />
            </div>
            </div>
        </details>

        {/* Configuration Parameters - Hidden for now */}
        <details className="mb-8 hidden">
            <summary className="text-md font-semibold text-primary font-system-mono cursor-pointer">Configuration</summary>
            <div className="ml-4 mt-4">
            <div className="grid grid-cols-1 gap-4">
                {/* Automation Interpolation Type */}
                <div className="space-y-2">
                    <label className="block text-sm font-medium text-foreground font-system-mono">
                        Automation Interpolation Type
                    </label>
                    <div className="text-xs text-gray-500 mb-2">
                        Type of automation interpolation
                    </div>
                    <select
                        value="linear"
                        disabled
                        className="w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-100 text-gray-600 cursor-not-allowed"
                    >
                        <option value="linear">Linear (model-supported)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                        Note: only linear automation interpolation is implemented in the model today.
                    </p>
                </div>

                {/* Taste Schedule Type */}
                <div className="space-y-2">
                    <label className="block text-sm font-medium text-foreground font-system-mono">
                        Taste Schedule Type
                    </label>
                    <div className="text-xs text-gray-500 mb-2">
                        Type of research taste schedule
                    </div>
                    <select
                        value={uiParameters.taste_schedule_type}
                        onChange={(e) => handleTasteScheduleChange(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="SDs per effective OOM">SDs per effective OOM</option>
                        <option value="SDs per progress-year">SDs per progress-year</option>
                    </select>
                </div>
            </div>
            </div>
        </details>

    </div>
    </>
    );
};
