import MarkdownRenderer from '@/components/MarkdownRenderer';
import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const markdown = await googleDocToMarkdown((await params).id);
  return (
    <div className="prose mx-auto p-6">
      <MarkdownRenderer markdown={markdown} />
      <hr />
      <div>
        <p>Renderer sanity test:</p>
        <MarkdownRenderer markdown={"x<sup>2</sup> + H<sub>2</sub>O"} />
      </div>
    </div>
  );
}


