import { useState, type ReactNode } from 'react';

interface WithChartTooltipProps {
  explanation: string;
  children: ReactNode;
  className?: string;
  fullWidth?: boolean;
  tooltipPlacement?: 'left' | 'right';
}

export const WithChartTooltip = ({
  explanation,
  children,
  className,
  fullWidth = false,
  tooltipPlacement = 'right',
}: WithChartTooltipProps) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const containerClasses = [
    'relative items-start gap-2 align-middle',
    fullWidth ? 'flex w-full justify-between' : 'inline-flex',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const labelClasses = fullWidth ? 'flex-1 leading-tight' : 'inline-flex leading-tight';
  const tooltipPositionClasses =
    tooltipPlacement === 'left'
      ? 'right-full mr-2 origin-top-right'
      : 'left-full ml-2 origin-top-left';

  return (
    <div className={containerClasses}>
      <span className={labelClasses}>{children}</span>
      <span
        className="mt-[0.5px] relative inline-flex h-4 w-4 min-w-4 items-center justify-center rounded-full border border-gray-400 text-[10px] font-semibold leading-none text-gray-600 cursor-help transition-colors hover:border-gray-600 hover:text-gray-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gray-500"
        role="button"
        tabIndex={0}
        aria-label="Show explanation"
        aria-expanded={showTooltip}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
      >
        i
        {showTooltip && (
          <div className={`pointer-events-none absolute top-full z-50 mt-1 ${tooltipPositionClasses}`}>
            <div className="w-64 rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-system-mono font-normal">
              {explanation}
            </div>
          </div>
        )}
      </span>
    </div>
  );
};