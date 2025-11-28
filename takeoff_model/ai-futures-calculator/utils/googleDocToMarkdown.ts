import { unified } from 'unified';
import rehypeParse from 'rehype-parse';
import rehypeRemark from 'rehype-remark';
import remarkGfm from 'remark-gfm';
import remarkStringify, { Options as RemarkStringifyOptions } from 'remark-stringify';
import juice from 'juice';
import rehypeGoogleStylesToSemantic from '@/utils/rehypeGoogleStylesToSemantic';
import rehypeStripFootnotes from '@/utils/rehypeStripFootnotes';
import rehypeTableCellNewlines from '@/utils/rehypeTableCellNewlines';
import { uploadGoogleDocImagesToCloudinary } from './cloudinary';
import type { Element as HastElement, Content as HastContent, Root as HastRoot, Text as HastText, Properties } from 'hast';
import type { Handle } from 'hast-util-to-mdast';

const serializeHeadingWithId: Handle = (_state, element) => element.properties?.id ? { type: 'html', value: serializeMinimalHtml(element) } : undefined;

const handlers: Record<string, Handle> = {
  // Preserve minimal HTML for sup/sub since remark-math does not cover them
  sup: (_state, element) => ({ type: 'html', value: serializeMinimalHtml(element) }),
  sub: (_state, element) => ({ type: 'html', value: serializeMinimalHtml(element) }),
  // Preserve <br> tags as HTML (especially important for table cells)
  br: () => ({ type: 'html', value: '<br>' }),
  // Preserve headings as HTML if they have an id attribute to enable internal links
  h1: serializeHeadingWithId,
  h2: serializeHeadingWithId,
  h3: serializeHeadingWithId,
  h4: serializeHeadingWithId,
  h5: serializeHeadingWithId,
  h6: serializeHeadingWithId,
};

export async function googleDocToMarkdown(docId: string): Promise<string> {
  const res = await fetch(`https://docs.google.com/document/d/${docId}/export?format=htm&tab=t.0`, { cache: 'force-cache', next: { revalidate: 60 } });
  if (!res.ok) {
    throw new Error(`Failed to fetch Google Doc: ${res.status}`);
  }

  const html = await res.text();
  const htmlWithCloudinaryImages = await uploadGoogleDocImagesToCloudinary(html);

  const inlined = juice(htmlWithCloudinaryImages);
  const stringifyOptions = { fences: true, bullet: '-' } as unknown as RemarkStringifyOptions & { allowDangerousHtml?: boolean };
  stringifyOptions.allowDangerousHtml = true;

  const file = await unified()
    .use(rehypeParse, { fragment: true })
    .use(rehypeGoogleStylesToSemantic)
    .use(rehypeStripFootnotes)
    .use(rehypeTableCellNewlines)
    .use(rehypeRemark, { handlers })
    .use(remarkGfm)
    .use(remarkStringify, stringifyOptions)
    .process(inlined);
  
  let markdown = String(file);
  
  // Fix: remarkStringify escapes underscores, but inside $$...$$ they should be literal
  // This is a brittle post-process to unescape them
  markdown = markdown.replace(/(\$\$[\s\S]+?\$\$)/g, (match) => match.replace(/\\_/g, '_'));
  markdown = markdown.replace(/(\$[^\$\n]+?\$)/g, (match) => match.replace(/\\_/g, '_'));
  
  return markdown;
}


function isText(node: HastContent | HastRoot | null | undefined): node is HastText {
  return !!node && (node as HastText).type === 'text';
}

function isElement(node: HastContent | HastRoot | null | undefined): node is HastElement {
  return !!node && (node as HastElement).type === 'element';
}

function serializeMinimalHtml(node: HastContent | HastRoot | null | undefined): string {
  if (!node) return '';
  if (isText(node)) {
    return String(node.value || '');
  }
  if (isElement(node)) {
    const tag = (node.tagName as string) || 'span';
    const id = node.properties?.id;
    const idAttr = id ? ` id="${String(id)}"` : '';
    const children = Array.isArray(node.children) ? node.children : [];
    const inner = children.map(serializeMinimalHtml).join('');
    return `<${tag}${idAttr}>${inner}</${tag}>`;
  }
  const children = ((node as unknown as { children?: HastContent[] }).children) || [];
  return children.map(serializeMinimalHtml).join('');
}
