import Link from "next/link";
import { googleDocToMarkdown } from "@/utils/googleDocToMarkdown";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { HeaderContent } from "@/components/HeaderContent";

const googleDocId = "1G2oxT1dIwHNggEwXYU_R4VYMANAxJRNb_KZPyRxWTIA";
const markdown = await googleDocToMarkdown(googleDocId);

const BodyMarkdown = <MarkdownRenderer markdown={markdown} />

export default function AboutPage() {
  return <>
    <HeaderContent variant="inline" className="pt-6 pb-4 px-6" />
    <main className="mt-10 prose mx-auto max-w-3xl">
      {BodyMarkdown}
    </main>
  </>
}
