"""
Parses raw SEC filing HTML into real sections (header -> list of paragraphs).

This logic went through significant real-world debugging (see README /
Day 3 notes for the full story): different filers structure their HTML
completely differently.
  - AAPL/TSLA: real headers are clean <b>/<strong> tags; body paragraphs
    live in <div> containers (no <p> tags in the document at all).
  - MSFT: headers are split across multiple sibling <span> tags inside
    a <p>, and body paragraphs DO use real <p> tags.

Both header detection and paragraph extraction use a dual-method
approach with automatic fallback, rather than assuming one HTML
convention applies to every filer.

Known, deliberately accepted limitations (see README):
  - Header titles for multi-span filers may contain a stray space at
    the split point (e.g. "ITEM 1. B USINESS" instead of "BUSINESS").
  - One isolated instance found (TSLA) of a section-boundary header not
    being detected, causing it to appear as trailing text in the
    previous section instead of starting a new one. Confirmed via
    targeted follow-up testing to be an isolated occurrence, not a
    systemic pattern.
"""

import re
from bs4 import BeautifulSoup


def is_bold(tag) -> bool:
    """A tag counts as bold if it's <b>/<strong>, or has inline
    font-weight >= 700 (CSS convention: 400=normal, 700=bold)."""
    if tag.name in ("b", "strong"):
        return True

    style = tag.get("style", "")
    if style:
        for decl in style.split(";"):
            if "font-weight" in decl:
                match = re.search(r"\d+", decl)
                if match and int(match.group()) >= 700:
                    return True
                if "bold" in decl.lower():
                    return True
    return False


def is_real_header(tag) -> bool:
    """Bold AND not nested inside a real hyperlink (i.e. not a TOC entry).
    A sibling anchor (used as a jump target) does NOT disqualify a header —
    only a *parent* <a> with a real href does (that's the TOC case)."""
    if not is_bold(tag):
        return False
    parent_link = tag.find_parent("a")
    if parent_link is not None and parent_link.get("href"):
        return False
    return True


def looks_like_item_header(text: str) -> bool:
    """Filter further: only keep bold text that actually matches the
    'Item 1A.' style pattern SEC filings use for real section titles.
    This avoids capturing random bold text (e.g. bolded emphasis in a
    sentence) that isn't a section header at all."""
    return bool(re.match(r"^Item\s+\d+[A-Z]?\.?\s", text.strip(), re.IGNORECASE))


def is_bold_paragraph(p_tag) -> bool:
    """A <p> counts as a bold header if ALL of its non-empty direct
    children are bold — handles headers split across multiple bold
    spans (e.g. 'ITEM 1A. RIS' + 'K FACTORS' as two sibling spans)."""
    children_with_text = [c for c in p_tag.find_all(True, recursive=False) if c.get_text(strip=True)]
    if not children_with_text:
        return is_bold(p_tag)
    return all(is_bold(child) for child in children_with_text)


def is_real_header_paragraph(p_tag) -> bool:
    """Bold paragraph AND not a TOC row (no real hyperlink wrapping or
    contained inside it)."""
    if not is_bold_paragraph(p_tag):
        return False
    parent_link = p_tag.find_parent("a")
    if parent_link is not None and parent_link.get("href"):
        return False
    inner_link = p_tag.find("a", href=True)
    if inner_link is not None:
        return False
    return True


def get_paragraphs_between(header, next_header) -> list[str]:
    """Collect paragraph-level text between two headers. Different filers
    use different tags for paragraphs:
      - AAPL/TSLA style: <div> containing <span> children (no <p> tags at all)
      - Other filers may use real <p> tags
    Tries <p> first; falls back to leaf-level <div> if none found.
    """
    # Attempt 1: real <p> tags
    paragraphs = []
    for element in header.next_elements:
        if element is next_header:
            break
        if getattr(element, "name", None) == "p":
            para_text = element.get_text(" ", strip=True)
            if para_text:
                paragraphs.append(para_text)

    if paragraphs:
        return paragraphs

    # Attempt 2: paragraph-like <div> — must be a "leaf" div (contains no
    # nested <div> of its own), otherwise we'd double-count text from both
    # a wrapper div and the paragraph divs nested inside it.
    for element in header.next_elements:
        if element is next_header:
            break
        if getattr(element, "name", None) == "div":
            style = element.get("style", "")
            if "display:none" in style.replace(" ", ""):
                continue
            if element.find("div"):
                # this div contains other divs -> it's a wrapper, skip it,
                # its inner divs will be visited separately by next_elements
                continue
            para_text = element.get_text(" ", strip=True)
            if para_text:
                paragraphs.append(para_text)

    return paragraphs


def split_into_sections(soup: BeautifulSoup) -> dict:
    """Main entry point: parse a filing's HTML into {section_title: [paragraphs]}."""
    # Try the original bold-tag method first (worked cleanly for AAPL/TSLA).
    all_bold = soup.find_all(["b", "strong"]) + [
        tag for tag in soup.find_all(True) if tag.get("style") and is_bold(tag) and tag.name not in ("b", "strong")
    ]
    headers = [tag for tag in all_bold if is_real_header(tag) and looks_like_item_header(tag.get_text(" ", strip=True))]

    # Fallback trigger: either too few headers found, OR the ones found look
    # truncated (a real "Item X. Title" is never this short — MSFT's split-span
    # bug produces things like "ITEM 1. B", which is a red flag, not a real title).
    MIN_EXPECTED_HEADERS = 5
    MIN_TITLE_LENGTH = 15
    suspicious = any(len(tag.get_text(" ", strip=True)) < MIN_TITLE_LENGTH for tag in headers)

    if len(headers) < MIN_EXPECTED_HEADERS or suspicious:
        reason = "too few headers" if len(headers) < MIN_EXPECTED_HEADERS else "truncated-looking titles"
        print(f"    [info] Bold-tag method looked unreliable ({reason}) — falling back to paragraph method.")
        all_paragraphs = soup.find_all("p")
        headers = [
            p for p in all_paragraphs
            if is_real_header_paragraph(p) and looks_like_item_header(p.get_text(" ", strip=True))
        ]

    # De-duplicate: sometimes the same header appears wrapped in multiple bold
    # tags right next to each other. Keep document order, drop consecutive dupes.
    seen_titles = set()
    unique_headers = []
    for tag in headers:
        title = tag.get_text(" ", strip=True)
        if title not in seen_titles:
            seen_titles.add(title)
            unique_headers.append(tag)
    headers = unique_headers

    sections = {}
    for i, header in enumerate(headers):
        title = header.get_text(" ", strip=True)
        next_header = headers[i + 1] if i + 1 < len(headers) else None

        sections[title] = get_paragraphs_between(header, next_header)

    return sections
