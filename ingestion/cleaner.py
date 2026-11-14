"""
Per-element text cleaning.

Each element_type gets its own cleaning rules on top of a shared
baseline pass (whitespace, unicode normalization, zero-width
character removal). Cleaning runs on the dict output of
ingestion/extractor.py, mutating only the "text" field.
"""

import re
import unicodedata
from collections import Counter
from typing import Any

ZERO_WIDTH_CHARS = "​‌‍﻿"


def clean_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply baseline + type-specific cleaning to every element in place."""
    for element in elements:
        element["text"] = _clean_baseline(element["text"])

    _strip_repeated_boilerplate(elements)

    for element in elements:
        element_type = element["element_type"]
        if element_type == "narrative" or element_type == "image_caption":
            element["text"] = _clean_narrative(element["text"])
        elif element_type == "table":
            element["text"] = _clean_table(element["text"])
        elif element_type == "list":
            element["text"] = _clean_list(element["text"])
        elif element_type == "code":
            element["text"] = _clean_code(element["text"])

    return elements


def _clean_baseline(text: str) -> str:
    """Strip whitespace, normalize unicode, and remove zero-width characters."""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")
    return text


def _strip_repeated_boilerplate(elements: list[dict[str, Any]]) -> None:
    """
    Remove short lines (<5 words) that repeat 3+ times across the document —
    these are typically running headers/footers that slipped past the
    Header/Footer element filter in extraction.
    """
    short_lines = [
        element["text"]
        for element in elements
        if element["element_type"] == "narrative" and len(element["text"].split()) < 5
    ]
    repeated = {line for line, count in Counter(short_lines).items() if count >= 3}
    if not repeated:
        return
    for element in elements:
        if element["element_type"] == "narrative" and element["text"] in repeated:
            element["text"] = ""


def _clean_narrative(text: str) -> str:
    """Collapse blank lines and rejoin hyphenated line breaks in prose text."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "some-\nword" -> "someword"
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _clean_table(text: str) -> str:
    """
    Convert a table element's plain text into a markdown table.
    Rows are assumed to be newline-separated, cells separated by
    runs of 2+ spaces or tab characters — Unstructured's default
    plain-text table rendering.
    """
    rows = [row for row in text.split("\n") if row.strip()]
    if not rows:
        return text

    table_rows = [re.split(r"\t+|\s{2,}", row.strip()) for row in rows]

    lines = ["| " + " | ".join(table_rows[0]) + " |"]
    lines.append("|" + "|".join(["------"] * len(table_rows[0])) + "|")
    for row in table_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _clean_list(text: str) -> str:
    """Prefix each list item with a bullet unless it already has one."""
    items = [item for item in text.split("\n") if item.strip()]
    bulleted = [
        item if re.match(r"^[•\-\*]\s", item.strip()) else f"• {item.strip()}"
        for item in items
    ]
    return "\n".join(bulleted)


def _clean_code(text: str) -> str:
    """Preserve code formatting exactly; wrap in triple backticks if missing."""
    if text.startswith("```") and text.endswith("```"):
        return text
    return f"```\n{text}\n```"
