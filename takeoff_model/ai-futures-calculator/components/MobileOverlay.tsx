export default function MobileOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-vivid-background sm:hidden">
      <div className="max-w-xs px-8 py-12 text-center">
        <div className="mb-6">
          <svg
            className="mx-auto h-16 w-16 text-vivid-mutedForeground"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25"
            />
          </svg>
        </div>
        <h2 className="mb-4 text-2xl font-semibold text-vivid-foreground">
          Desktop Only
        </h2>
        <p className="text-vivid-mutedForeground leading-relaxed">
          This AI progress modeling tool requires a desktop browser for the best experience.
          Please visit us on a larger screen to access all features and visualizations.
        </p>
      </div>
    </div>
  );
}