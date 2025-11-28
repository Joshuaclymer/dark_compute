const SUPERSCRIPT_MAP: Record<string, string> = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '-': '⁻', '+': '⁺'
};

function roundPowerOfTenExponent(value: number): number {
    if (!Number.isFinite(value) || value === 0) {
        return 0;
    }

    const absVal = Math.abs(value);

    if (absVal < 10) {
        return Math.round(value);
    }

    const digits = Math.floor(Math.log10(absVal)) + 1;
    const factor = Math.pow(10, Math.max(digits - 2, 0));
    const rounded = Math.round(value / factor) * factor;

    return rounded;
}

export type FormatPowerOfTenNodeOptions = {
    suffix?: React.ReactNode;
    mantissaOpacity?: number;
    mantissaFontSize?: string | number;
    exponentFontSize?: string | number;
    renderMode?: 'html' | 'svg';
    baselineShift?: `0.${number}em`;
};

export function formatPowerOfTenNode(value: number, options: FormatPowerOfTenNodeOptions = {}): React.ReactNode {
    if (!Number.isFinite(value)) {
        return null;
    }

    const {
        suffix,
        mantissaOpacity = 0.7,
        mantissaFontSize = '0.92em',
        exponentFontSize,
        renderMode = 'html',
        baselineShift = '0.2em',
    } = options;

    const exponent = roundPowerOfTenExponent(value);

    if (exponent === 0) {
        if (renderMode === 'svg') {
            return (
                <>
                    <tspan>1</tspan>
                    {suffix != null ? <tspan>{suffix}</tspan> : null}
                </>
            );
        }

        return (
            <span>
                1
                {suffix}
            </span>
        );
    }

    if (renderMode === 'svg') {
        return (
            <>
                <tspan fillOpacity={mantissaOpacity} fontSize={mantissaFontSize}>10</tspan>
                <tspan baselineShift={baselineShift} fontSize={exponentFontSize}>{exponent}</tspan>
                {suffix != null ? <tspan>{suffix}</tspan> : null}
            </>
        );
    }

    return (
        <span>
            <span style={{ opacity: mantissaOpacity, fontSize: mantissaFontSize }}>10</span>
            <sup style={exponentFontSize ? { fontSize: exponentFontSize } : undefined}>{exponent}</sup>
            {suffix}
        </span>
    );
}

export function formatPowerOfTenText(value: number, options: { suffix?: string } = {}): string {
    if (!Number.isFinite(value)) {
        return '';
    }

    const exponent = roundPowerOfTenExponent(value);

    if (exponent === 0) {
        return `1${options.suffix ?? ''}`;
    }

    const exponentStr = exponent.toString();
    const superscript = exponentStr.split('').map(ch => SUPERSCRIPT_MAP[ch] ?? ch).join('');

    return `10${superscript}${options.suffix ?? ''}`;
}

export function formatAsPowerOfTenNode(value: number, options: FormatPowerOfTenNodeOptions = {}): React.ReactNode {
    if (!Number.isFinite(value) || value <= 0) {
        return null;
    }

    return formatPowerOfTenNode(Math.log10(value), options);
}

export function formatAsPowerOfTenText(value: number, options: { suffix?: string } = {}): string {
    if (!Number.isFinite(value) || value <= 0) {
        return '';
    }

    return formatPowerOfTenText(Math.log10(value), options);
}

export function formatTo3SigFigs(value: number): string {
    if (value === 0) return '0';

    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    // Use superscript 10^x for very small or very large values
    if (absValue < 0.001 || absValue >= 1e6) {
        const rawExponent = Math.floor(Math.log10(absValue));
        // Round exponent to 1 significant figure
        const order = Math.floor(Math.log10(Math.abs(rawExponent)));
        const scale = Math.pow(10, order);
        const roundedExponent = Math.round(rawExponent / scale) * scale;
        if (roundedExponent === 0) return sign + '1';

        const exponentStr = roundedExponent.toString();
        const superscript = exponentStr.split('').map(ch => SUPERSCRIPT_MAP[ch] ?? ch).join('');
        return sign + '10' + superscript;
    }

    // For normal range values, round to 3 significant figures
    const magnitude = Math.floor(Math.log10(absValue));
    const scale = Math.pow(10, 2 - magnitude);
    const rounded = Math.round(absValue * scale) / scale;

    // Convert to string and remove trailing zeros after decimal
    let result = rounded.toString();
    if (result.includes('.')) {
        result = result.replace(/\.?0+$/, '');
    }

    return sign + result;
}

export function yearsToMinutes(years: number): number {
    return years * 525600; // 365 days * 24 hours * 60 minutes
}

function formatUnitValue(value: number, unit: string, fractionDigits: number): string {
    const fixed = value.toFixed(fractionDigits);
    const trimmed = fixed.includes('.')
        ? fixed.replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.0+$/, '')
        : fixed;
    return `${trimmed} ${unit}`;
}

export function formatTimeDuration(minutes: number): string {
    if (minutes < 0.0167) {
        return formatUnitValue(minutes * 60, 'sec', 2);
    } else if (minutes < 1) {
        const seconds = minutes * 60;
        return Math.abs(seconds - 1) < 0.01 ? '1 sec' : formatUnitValue(seconds, 'sec', 1);
    } else if (minutes < 60) {
        return formatUnitValue(minutes, 'min', 1);
    } else if (minutes < 1440) {
        return formatUnitValue(minutes / 60, 'hr', 1);
    } else if (minutes < 43200) {
        return formatUnitValue(minutes / 1440, 'days', 1);
    } else if (minutes < 525600) {
        return formatUnitValue(minutes / 43200, 'months', 1);
    } else {
        return formatUnitValue(minutes / 525600, 'years', 1);
    }
}

export function formatWorkTimeDuration(minutes: number): string {
    if (minutes < 0.0167) {
        return formatUnitValue(minutes * 60, 'sec', 2);
    } else if (minutes < 1) {
        const seconds = minutes * 60;
        return Math.abs(seconds - 1) < 0.01 ? '1 sec' : formatUnitValue(seconds, 'sec', 1);
    } else if (minutes < 60) {
        return formatUnitValue(minutes, 'min', 1);
    } else if (minutes < 1440) {
        return formatUnitValue(minutes / 60, 'hr', 1);
    } else if (minutes < 43200) {
        const workDayMinutes = 8 * 60;
        return formatUnitValue(minutes / workDayMinutes, 'work days', 1);
    } else if (minutes < 525600) {
        const workMonthMinutes = (2000 / 12) * 60;
        return formatUnitValue(minutes / workMonthMinutes, 'work months', 1);
    } else {
        const workYearMinutes = 2000 * 60;
        return formatUnitValue(minutes / workYearMinutes, 'work years', 1);
    }
}

export function formatSCHorizon(logValue: number): string {
    const minutes = Math.pow(10, logValue);
    return formatWorkTimeDuration(minutes).replace(/\.0\s/, ' '); // Remove .0 for cleaner display
}

export function formatTimeDurationDetailed(minutes: number): string {
    const MINUTES_PER_YEAR = 525600;
    const MINUTES_PER_MONTH = 43800;
    const MINUTES_PER_DAY = 1440;
    const MINUTES_PER_HOUR = 60;

    const formatWithUnit = (value: number, unit: string): string => {
        const floored = Math.floor(value * 10) / 10;
        return `${floored.toFixed(1)} ${unit}`;
    };

    // Handle extremely large values (more than 10 billion years)
    if (minutes >= MINUTES_PER_YEAR * 1e10) {
        const years = minutes / MINUTES_PER_YEAR;
        // Use scientific notation for extremely large year values
        return `${formatTo3SigFigs(years)} years`;
    }

    if (minutes >= MINUTES_PER_YEAR) {
        const years = minutes / MINUTES_PER_YEAR;
        return formatWithUnit(years, 'years');
    }

    if (minutes >= MINUTES_PER_MONTH) {
        const months = minutes / MINUTES_PER_MONTH;
        return formatWithUnit(months, 'months');
    }

    if (minutes >= MINUTES_PER_DAY) {
        const days = minutes / MINUTES_PER_DAY;
        return formatWithUnit(days, 'days');
    }

    if (minutes >= MINUTES_PER_HOUR) {
        const hours = minutes / MINUTES_PER_HOUR;
        return formatWithUnit(hours, 'hours');
    }

    if (minutes >= 1) {
        return formatWithUnit(minutes, 'min');
    }

    // Handle sub-minute durations
    const seconds = minutes * 60;
    return formatWithUnit(seconds, 'sec');
}

export function formatWorkTimeDurationDetailed(minutes: number): string {
    const MINUTES_PER_WORK_YEAR = 2000 * 60;
    const MINUTES_PER_WORK_MONTH = (2000 / 12) * 60;
    const MINUTES_PER_WORK_DAY = 8 * 60;
    const MINUTES_PER_HOUR = 60;

    const formatWithUnit = (value: number, unit: string): string => {
        const floored = Math.floor(value * 10) / 10;
        return `${floored.toFixed(1)} ${unit}`;
    };

    // Handle extremely large values (more than 10 billion years)
    if (minutes >= MINUTES_PER_WORK_YEAR * 1e10) {
        const years = minutes / MINUTES_PER_WORK_YEAR;
        // Use scientific notation for extremely large year values
        return `${formatTo3SigFigs(years)} years`;
    }

    if (minutes >= MINUTES_PER_WORK_YEAR) {
        const years = minutes / MINUTES_PER_WORK_YEAR;
        return formatWithUnit(years, 'years');
    }

    if (minutes >= MINUTES_PER_WORK_MONTH) {
        const months = minutes / MINUTES_PER_WORK_MONTH;
        return formatWithUnit(months, 'months');
    }

    if (minutes >= MINUTES_PER_WORK_DAY) {
        const days = minutes / MINUTES_PER_WORK_DAY;
        return formatWithUnit(days, 'days');
    }

    if (minutes >= MINUTES_PER_HOUR) {
        const hours = minutes / MINUTES_PER_HOUR;
        return formatWithUnit(hours, 'hours');
    }

    if (minutes >= 1) {
        return formatWithUnit(minutes, 'min');
    }

    // Handle sub-minute durations
    const seconds = minutes * 60;
    return formatWithUnit(seconds, 'sec');
}


export function formatLogWorkTick(tickItem: number, workTimeLabel?: boolean): string {
    // Convert minutes to appropriate unit with 3 sig figs
    if (tickItem < 1) {
        const seconds = tickItem * 60;
        return `${formatTo3SigFigs(seconds)} sec`;
    } else if (tickItem < 60) {
        return `${formatTo3SigFigs(tickItem)} min`;
    } else if (tickItem < 1440) {
        const hours = tickItem / 60;
        return `${formatTo3SigFigs(hours)} hr`;
    } else if (tickItem < (43200 / 3)) {
        const days = tickItem / (8 * 60);
        return `${formatTo3SigFigs(days)} ${workTimeLabel ? 'work days' : 'days'}`;
    } else if (tickItem < 525600) {
        const months = tickItem / ((2000 / 12) * 60);
        return `${formatTo3SigFigs(months)} ${workTimeLabel ? 'work months' : 'months'}`;
    } else if (tickItem < 5256000) {
        const years = tickItem / (2000 * 60);
        return `${formatTo3SigFigs(years)} ${workTimeLabel ? 'work years' : 'years'}`;
    } else if (tickItem < 5256000000) {
        const millennia = tickItem / ((2000 * 60) * 1000);
        return `${formatTo3SigFigs(millennia)} ${workTimeLabel ? 'work millennia' : 'millennia'}`;
    } else {
        return '10 millennia';
    }
}

export function formatEffectiveComputeValue(value: number): React.ReactNode {
    return formatPowerOfTenNode(value);
}

export function formatOOMSuperscript(value: number): React.ReactNode {
    return formatPowerOfTenNode(value, { suffix: ' FLOPs' });
}

export function formatOOMSuperscriptText(value: number): string {
    return formatPowerOfTenText(value);
}

export function formatOOMSuperscriptNode(value: number): React.ReactNode {
    return formatPowerOfTenNode(value);
}

export function formatYearMonth(decimalYear: number): string {
    const year = Math.floor(decimalYear);
    const fraction = decimalYear - year;
    const monthIndexRaw = Math.round(fraction * 12);

    // Handle edge case where rounding gives us month 12 (should roll to next year)
    const monthIndex = monthIndexRaw % 12;
    const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

    const months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];

    return `${months[monthIndex]} ${adjustedYear}`;
}

export function formatYearMonthShort(decimalYear: number): string {
    if (!Number.isFinite(decimalYear)) {
        return '';
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    const baseYear = Math.floor(decimalYear);
    const fraction = decimalYear - baseYear;
    const rawMonths = Math.round(fraction * 12);
    const yearAdjustment = Math.floor(rawMonths / 12);
    const normalizedMonthIndex = ((rawMonths % 12) + 12) % 12;
    const year = baseYear + yearAdjustment;

    const monthLabel = months[normalizedMonthIndex];

    return `${monthLabel} ${year}`;
}
