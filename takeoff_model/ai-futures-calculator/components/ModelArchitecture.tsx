'use client';

import ModelDiagram from '@/svgs/model-diagram-semantic.svg';
import { useCallback, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import explanationsHtml from '@/public/explanations.html';
import { useParameterHover, PARAMETER_TO_SVG_NODES } from './ParameterHoverContext';

type SectionEntry = {
  label: string;
  x: number | null;
};

const normalizeLabel = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const normalized = input.replace(/\s+/g, ' ').trim();
  return normalized.length > 0 ? normalized : null;
};

const normalizeHeadingText = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const withoutPrefix = input.replace(/^[\d\s.:()-]+/, '');
  return normalizeLabel(withoutPrefix);
};

const buildMatchKey = (input: string | null | undefined) => {
  const normalized = normalizeLabel(input);
  if (!normalized) {
    return null;
  }
  return normalized.toLowerCase().replace(/[^a-z0-9]+/g, '');
};

const parseXCoordinate = (element: Element | null) => {
  if (!element) {
    return null;
  }
  const rawValue = element.getAttribute('x');
  if (!rawValue) {
    return null;
  }
  const parsed = Number.parseFloat(rawValue);
  return Number.isNaN(parsed) ? null : parsed;
};

const findNearestSectionLabel = (sections: SectionEntry[], nodeX: number | null) => {
  if (sections.length === 0) {
    return null;
  }
  if (nodeX == null) {
    return sections[0].label;
  }
  const nearest = sections.reduce<SectionEntry>((best, candidate) => {
    if (candidate.x == null) {
      return best;
    }
    if (best.x == null) {
      return candidate;
    }
    const bestDistance = Math.abs(best.x - nodeX);
    const candidateDistance = Math.abs(candidate.x - nodeX);
    return candidateDistance < bestDistance ? candidate : best;
  }, sections[0]);

  return nearest.label;
};

const findForeignObject = (element: Element | null) => {
  let current: Element | null = element;
  while (current && current.tagName.toLowerCase() !== 'foreignobject') {
    current = current.parentElement;
  }

  return current;
};

type DiagramLabelElement = HTMLElement | SVGTextElement;

const isSvgTextElement = (element: Element): element is SVGTextElement =>
  element.tagName.toLowerCase() === 'text';

const isTspanElement = (element: Element): element is SVGTSpanElement =>
  element.tagName.toLowerCase() === 'tspan';

const isParagraphSection = (element: HTMLElement) =>
  element.style.fontStyle === 'italic' && element.style.fontSize === '48px';

const TEXT_FONT_THRESHOLD = 12;

const shouldIncludeSvgTextElement = (element: SVGTextElement) => {
  const label = normalizeLabel(element.textContent);
  if (!label) {
    return false;
  }
  const fontSizeAttr = element.getAttribute('font-size');
  const fontSize = fontSizeAttr ? Number.parseFloat(fontSizeAttr) : Number.NaN;
  if (!Number.isNaN(fontSize) && fontSize < TEXT_FONT_THRESHOLD) {
    return false;
  }
  return true;
};

const getLabelElements = (svg: SVGElement): DiagramLabelElement[] => {
  const paragraphElements = Array.from(svg.querySelectorAll('foreignObject p')) as HTMLElement[];
  const textElements = Array.from(svg.querySelectorAll('text')).filter((text): text is SVGTextElement =>
    isSvgTextElement(text) && shouldIncludeSvgTextElement(text),
  );
  return [...paragraphElements, ...textElements];
};

const getElementXCoordinate = (element: DiagramLabelElement): number | null => {
  if (element instanceof SVGGraphicsElement) {
    try {
      const bbox = element.getBBox();
      return Number.isFinite(bbox.x) ? bbox.x : null;
    } catch {
      return null;
    }
  }
  const foreignObject = findForeignObject(element);
  return parseXCoordinate(foreignObject);
};

const ensureOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    if (!element.hasAttribute('data-original-fill')) {
      const fill = element.getAttribute('fill') ?? '#1E1E1E';
      element.setAttribute('data-original-fill', fill);
    }
  } else if (!element.hasAttribute('data-original-color')) {
    const computedStyle = window.getComputedStyle(element);
    element.setAttribute('data-original-color', computedStyle.color);
  }
};

const restoreOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    const originalFill = element.getAttribute('data-original-fill');
    if (originalFill) {
      element.setAttribute('fill', originalFill);
    }
  } else {
    const originalColor = element.getAttribute('data-original-color');
    if (originalColor) {
      element.style.color = originalColor;
    }
  }
};

const setTextHighlightColor = (element: DiagramLabelElement, color: string) => {
  if (isSvgTextElement(element)) {
    element.setAttribute('fill', color);
  } else {
    element.style.color = color;
  }
};

const findLabelElement = (target: EventTarget | null): DiagramLabelElement | null => {
  let current: Element | null = null;
  if (target instanceof Element) {
    current = target;
  } else if (target instanceof Node) {
    current = target.parentElement;
  }

  while (current) {
    const tagName = current.tagName.toLowerCase();
    if (tagName === 'p' || tagName === 'text') {
      return current as DiagramLabelElement;
    }
    if (tagName === 'tspan') {
      current = current.parentElement;
      continue;
    }
    current = current.parentElement;
  }

  return null;
};

type PointerPosition = {
  x: number;
  y: number;
};

const MAX_POINTER_DISTANCE = 140;

const getNearestLabelElement = (svg: SVGElement, pointer: PointerPosition): DiagramLabelElement | null => {
  const labelElements = getLabelElements(svg);
  let closestElement: DiagramLabelElement | null = null;
  let closestDistance = Number.POSITIVE_INFINITY;

  labelElements.forEach((element) => {
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      return;
    }
    const clampedX = Math.min(Math.max(pointer.x, rect.left), rect.right);
    const clampedY = Math.min(Math.max(pointer.y, rect.top), rect.bottom);
    const dx = pointer.x - clampedX;
    const dy = pointer.y - clampedY;
    const distance = Math.hypot(dx, dy);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestElement = element;
    }
  });

  if (closestDistance <= MAX_POINTER_DISTANCE) {
    return closestElement;
  }
  return null;
};

const buildSectionEntries = (svg: SVGElement): SectionEntry[] =>
  Array.from(svg.querySelectorAll('foreignObject')).reduce<SectionEntry[]>(
    (acc, foreignObject) => {
      const paragraph = foreignObject.querySelector('p');
      if (!paragraph) {
        return acc;
      }
      const label = normalizeLabel(paragraph.textContent);
      if (!label) {
        return acc;
      }

      const paragraphElement = paragraph as HTMLElement;
      const isSection =
        paragraphElement.style.fontStyle === 'italic' &&
        paragraphElement.style.fontSize === '48px';

      if (!isSection) {
        return acc;
      }

      acc.push({
        label,
        x: parseXCoordinate(foreignObject),
      });
      return acc;
    },
    [],
  );

const hasValidTarget = (label: string | null, headingLookup: Map<string, HTMLElement>): boolean => {
  const key = buildMatchKey(label);
  if (!key) {
    return false;
  }

  let target = headingLookup.get(key);
  if (!target) {
    for (const [storedKey, element] of headingLookup.entries()) {
      if (storedKey.includes(key) || key.includes(storedKey)) {
        target = element;
        break;
      }
    }
  }

  return target != null;
};

export default function ModelArchitecture() {
  const explanationContainerRef = useRef<HTMLDivElement | null>(null);
  const headingLookupRef = useRef<Map<string, HTMLElement>>(new Map());
  const svgContainerRef = useRef<HTMLDivElement | null>(null);
  const [isHoveringSvg, setIsHoveringSvg] = useState(false);
  const { hoveredParameter } = useParameterHover();

  useEffect(() => {
    const container = explanationContainerRef.current;
    const nextLookup = new Map<string, HTMLElement>();

    if (container) {
      const headings = container.querySelectorAll('h3');
      headings.forEach((heading) => {
        const normalizedHeading = normalizeHeadingText(heading.textContent);
        const key = buildMatchKey(normalizedHeading);
        if (!key || nextLookup.has(key)) {
          return;
        }
        const element = heading as HTMLElement;
        if (!element.id) {
          element.id = `section-${key}`;
        }
        if (!element.hasAttribute('tabindex')) {
          element.setAttribute('tabindex', '-1');
        }
        element.style.scrollMarginTop = '30rem';
        nextLookup.set(key, element);
      });
    }

    headingLookupRef.current = nextLookup;
  }, []);

  const scrollToHeading = useCallback((label: string | null) => {
    const key = buildMatchKey(label);
    if (!key) {
      return;
    }

    const lookup = headingLookupRef.current;
    let target = lookup.get(key);
    if (!target) {
      for (const [storedKey, element] of lookup.entries()) {
        if (storedKey.includes(key) || key.includes(storedKey)) {
          target = element;
          break;
        }
      }
    }

    if (!target) {
      return;
    }

    target.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const handleMouseEnter = () => {
      setIsHoveringSvg(true);
    };

    const handleMouseLeave = () => {
      setIsHoveringSvg(false);
    };

    svg.addEventListener('mouseenter', handleMouseEnter);
    svg.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      svg.removeEventListener('mouseenter', handleMouseEnter);
      svg.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const headingLookup = headingLookupRef.current;
    const sectionEntries = buildSectionEntries(svg);
    const labelElements = getLabelElements(svg);

    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        return;
      }

      ensureOriginalColor(element);

      const isSection =
        element instanceof HTMLElement ? isParagraphSection(element) : false;

      const nodeX = getElementXCoordinate(element);
      const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);
      const nodeLabel = isSection ? null : label;
      const targetLabel = nodeLabel ?? sectionLabel;

      const isValidLink = hasValidTarget(targetLabel, headingLookup) && !isSection;

      if (isHoveringSvg && isValidLink) {
        element.style.textDecorationStyle = 'dotted';
        element.style.textDecorationColor = '#2A623D';
        element.style.textDecorationThickness = '3px';
        element.style.cursor = 'pointer';
      } else {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        restoreOriginalColor(element);
      }
    });
  }, [isHoveringSvg]);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const highlightLabels = hoveredParameter ? PARAMETER_TO_SVG_NODES[hoveredParameter] : null;

    const labelElements = getLabelElements(svg);
    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        return;
      }

      ensureOriginalColor(element);

      const shouldHighlight = highlightLabels?.includes(label) ?? false;

      if (shouldHighlight) {
        if (element instanceof HTMLElement) {
          element.style.color = '#8b0000';
          element.style.fontWeight = 'bold';
          element.style.padding = '2px 4px';
          element.style.borderRadius = '3px';
          element.style.transition = 'all 0.15s ease-in-out';
        } else {
          element.style.fontWeight = 'bold';
          setTextHighlightColor(element, '#8b0000');
        }
      } else {
        if (element instanceof HTMLElement) {
          element.style.backgroundColor = '';
          element.style.fontWeight = '';
          element.style.padding = '';
          element.style.borderRadius = '';
          restoreOriginalColor(element);
        } else {
          element.style.fontWeight = '';
          restoreOriginalColor(element);
        }
      }
    });
  }, [hoveredParameter]);

  const navigateFromSvgTarget = useCallback((eventTarget: EventTarget | null, svg: SVGElement, pointer?: PointerPosition) => {
    let labelElement = findLabelElement(eventTarget);
    if (!labelElement && pointer) {
      labelElement = getNearestLabelElement(svg, pointer);
    }
    if (!labelElement) {
      return;
    }

    const label = normalizeLabel(labelElement.textContent);
    if (!label) {
      return;
    }

    const nodeX = getElementXCoordinate(labelElement);

    const isSection =
      labelElement instanceof HTMLElement ? isParagraphSection(labelElement) : false;

    const sectionEntries = buildSectionEntries(svg);

    const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);

    const nodeLabel = isSection ? null : label;
    scrollToHeading(nodeLabel ?? sectionLabel);

    console.log('model-architecture-click', {
      node: nodeLabel ?? undefined,
      section: sectionLabel ?? undefined,
    });
  }, [scrollToHeading]);

  const handleSvgClick = useCallback((event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();

    const svg = event.currentTarget as SVGElement | null;
    if (!svg) {
      return;
    }

    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  const handleContainerClick = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }
    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  useEffect(() => {
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }

    svg.addEventListener('click', handleSvgClick);
    return () => {
      svg.removeEventListener('click', handleSvgClick);
    };
  }, [handleSvgClick]);

  return (
    <div className="mt-5">
      <div className="sticky top-0 self-start bg-vivid-background pb-4 pt-4 border-b border-gray-500/30 w-full -ml-6" onClick={handleContainerClick}>
        <div ref={svgContainerRef}>
          <ModelDiagram
            id="model-architecture-diagram-svg"
            className="mx-auto"
          />
        </div>
      </div>
      <div
        ref={explanationContainerRef}
        className="prose max-w-3xl mx-auto scroll-p-20"
        dangerouslySetInnerHTML={{ __html: explanationsHtml }}
      />
    </div>
  );
}
