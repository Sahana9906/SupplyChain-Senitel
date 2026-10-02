"""Splits document text into overlapping chunks small enough to embed
meaningfully. Breaks on whitespace where possible rather than mid-word.
Pure text processing -- no LLM call, deterministic.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            break_point = text.rfind(" ", start, end)
            if break_point <= start:
                break_point = end
        else:
            break_point = end

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        if break_point >= n:
            break

        next_start = break_point - overlap
        if next_start <= start:
            next_start = break_point
        else:
            space_pos = text.find(" ", next_start)
            if space_pos == -1 or space_pos >= break_point:
                next_start = break_point
            else:
                next_start = space_pos + 1
        start = next_start

    return chunks


def chunk_by_sections(text: str, chunk_size: int = 800, overlap: int = 150) -> list[tuple[str | None, str]]:
    """Splits on markdown "## " headings first, then applies chunk_text
    within each section -- so a chunk from a contract's "Penalty Terms"
    section carries that section name, not just an arbitrary character
    offset. Returns (section_title_or_None, chunk_text) pairs.

    Content before the first "## " heading (the title + any preamble) is
    treated as its own section with section_title=None.
    """
    import re

    text = text.strip()
    if not text:
        return []

    heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [(None, c) for c in chunk_text(text, chunk_size, overlap)]

    sections: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))

    result: list[tuple[str | None, str]] = []
    for title, body in sections:
        for chunk in chunk_text(body, chunk_size, overlap):
            result.append((title, chunk))
    return result
