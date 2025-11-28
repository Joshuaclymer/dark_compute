import Link from "next/link";

export const HeaderContent = ({ variant, className = '' }: { variant: 'inline' | 'floating'; className?: string }) => (
  <header className={`flex flex-row items-center justify-between text-left ${className}`}>
    <h1 className={`text-3xl font-bold text-primary font-et-book text-left ${variant === 'inline' ? 'mb-2' : ''}`}>
      <Link href="/">AI Futures Timelines & Takeoff Model</Link>
    </h1>
    <div className="flex flex-row gap-10 mr-10 text-sm">
      <Link href="/#model-explanation" className="font-system-mono text-xs">
        Model Explanation
      </Link>
      <Link href="/about" className="font-system-mono text-xs">
        About
      </Link>
      <Link href="/forecast" className="font-system-mono text-xs">
        Forecast
      </Link>
    </div>
  </header>
);
