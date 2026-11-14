"""
Element-aware document parsing via Unstructured.io.

Documents are never treated as flat text — each Unstructured element
is classified by type (narrative, table, list, code, image caption)
and tagged with the section heading it falls under, so downstream
cleaning and chunking can apply type-specific rules.
"""

from typing import Any

from unstructured.documents.elements import (
    CodeSnippet,
    Footer,
    Header,
    Image,
    ListItem,
    PageBreak,
    Table,
    Title,
)
from unstructured.partition.auto import partition


def extract_elements(file_path: str) -> list[dict[str, Any]]:
    """Parse a document into a list of classified, section-tagged elements."""
    elements = partition(file_path)
    parsed: list[dict[str, Any]] = []
    current_section = "Introduction"  # default heading until a Title is seen

    for element in elements:

        # Titles update the running section heading but are not stored
        # as content themselves — they become metadata on later elements.
        if isinstance(element, Title):
            current_section = element.text.strip()
            continue

        # Headers, footers, and page breaks carry no retrievable content.
        if isinstance(element, (Header, Footer, PageBreak)):
            continue

        # Skip empty or near-empty elements (e.g. stray whitespace artifacts).
        if not element.text or len(element.text.strip()) < 10:
            continue

        parsed.append(
            {
                "text": element.text.strip(),
                "element_type": classify_element(element),
                "section": current_section,
                "page_number": element.metadata.page_number
                if hasattr(element.metadata, "page_number")
                else None,
            }
        )

    return parsed


def classify_element(element: Any) -> str:
    """Map an Unstructured element class to our internal element_type tag."""
    if isinstance(element, Table):
        return "table"
    if isinstance(element, ListItem):
        return "list"
    if isinstance(element, CodeSnippet):
        return "code"
    if isinstance(element, Image):
        return "image_caption"
    return "narrative"
