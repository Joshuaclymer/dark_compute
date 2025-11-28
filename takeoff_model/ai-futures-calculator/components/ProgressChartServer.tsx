import { Suspense } from 'react';
import { loadBenchmarkData } from '../utils/benchmarkLoader';
import ProgressChartClient from './ProgressChartClient';
import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';
import MarkdownRenderer from './MarkdownRenderer';

const GOOGLE_DOC_ID = '1_Fe34EcaYP5xLXtcfydZYH9-8-0zpSB2_DilV-0CwQM';

const markdown = await googleDocToMarkdown(GOOGLE_DOC_ID);

const IntroMarkdown = <MarkdownRenderer markdown={markdown} />

export default function ProgressChartServer() {
  const benchmarkData = loadBenchmarkData();

  return (
    <Suspense
      fallback={(
        <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
          Loading model...
        </div>
      )}
    >
      <ProgressChartClient benchmarkData={benchmarkData} IntroductionMarkdown={IntroMarkdown} />
    </Suspense>
  );
}
