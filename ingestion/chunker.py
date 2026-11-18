"""
Element-aware hybrid chunking.

Tables and code blocks are never split. Lists are kept whole when they
fit, otherwise split at item boundaries. Narrative text (and image
captions) use recursive character splitting with token overlap.
"""

from typing import Any

import tiktoken

MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 50
NARRATIVE_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
BLOCK_SEPARATORS = ["\n\n", "\n", " ", ""]

# text-embedding-3-small hard-rejects any single input over 8192 tokens.
# "Never split" tables/code is a preference, not a guarantee — a big
# enough table or code block must still be split, or ingestion fails
# outright with an OpenAI 400 on the whole batch.
MAX_UNSPLIT_BLOCK_TOKENS = 7500

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding shared across the app."""
    return len(_encoder.encode(text))


def chunk_elements(
    elements: list[dict[str, Any]], doc_name: str, doc_hash: str
) -> list[dict[str, Any]]:
    """Chunk a list of cleaned elements into retrieval-ready chunks."""
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for element in elements:
        element_type = element["element_type"]
        text = element["text"]
        if not text:
            continue
        token_count = count_tokens(text)

        # --- TABLE / CODE: kept whole, unless that would exceed the
        # embedding model's hard input limit — then split as a last resort.
        if element_type in ("table", "code"):
            if token_count <= MAX_UNSPLIT_BLOCK_TOKENS:
                chunks.append(
                    build_chunk(text, element_type, chunk_index, element, doc_name, doc_hash)
                )
                chunk_index += 1
            else:
                sub_chunks = recursive_split(
                    text=text,
                    separators=BLOCK_SEPARATORS,
                    max_tokens=MAX_UNSPLIT_BLOCK_TOKENS,
                    overlap_tokens=0,
                )
                for sub in sub_chunks:
                    if not sub.strip():
                        continue
                    chunks.append(
                        build_chunk(sub, element_type, chunk_index, element, doc_name, doc_hash)
                    )
                    chunk_index += 1

        # --- LIST: keep together if it fits, else split at item boundaries ---
        elif element_type == "list":
            if token_count <= MAX_CHUNK_TOKENS:
                chunks.append(
                    build_chunk(text, "list", chunk_index, element, doc_name, doc_hash)
                )
                chunk_index += 1
            else:
                for sub in _split_list_by_items(text):
                    chunks.append(
                        build_chunk(sub, "list", chunk_index, element, doc_name, doc_hash)
                    )
                    chunk_index += 1

        # --- NARRATIVE / IMAGE CAPTION: recursive character chunking ---
        elif element_type in ("narrative", "image_caption"):
            if token_count <= MAX_CHUNK_TOKENS:
                chunks.append(
                    build_chunk(text, element_type, chunk_index, element, doc_name, doc_hash)
                )
                chunk_index += 1
            else:
                sub_chunks = recursive_split(
                    text=text,
                    separators=NARRATIVE_SEPARATORS,
                    max_tokens=MAX_CHUNK_TOKENS,
                    overlap_tokens=OVERLAP_TOKENS,
                )
                for sub in sub_chunks:
                    if count_tokens(sub) < MIN_CHUNK_TOKENS:
                        continue  # skip tiny trailing fragments
                    chunks.append(
                        build_chunk(
                            sub, element_type, chunk_index, element, doc_name, doc_hash
                        )
                    )
                    chunk_index += 1

    return chunks


def _split_list_by_items(text: str) -> list[str]:
    """Pack list items into <=500-token groups without breaking an item."""
    items = text.split("\n")
    groups: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for item in items:
        item_tokens = count_tokens(item)

        # A single item larger than the embedding model's hard limit
        # (a huge pasted code block as one "bullet", say) still has to
        # be split further — it can't be packed as one atomic piece.
        if item_tokens > MAX_UNSPLIT_BLOCK_TOKENS:
            if buffer:
                groups.append("\n".join(buffer))
                buffer, buffer_tokens = [], 0
            groups.extend(
                recursive_split(
                    text=item,
                    separators=BLOCK_SEPARATORS,
                    max_tokens=MAX_UNSPLIT_BLOCK_TOKENS,
                    overlap_tokens=0,
                )
            )
            continue

        if buffer_tokens + item_tokens > MAX_CHUNK_TOKENS:
            if buffer:
                groups.append("\n".join(buffer))
            buffer = [item]
            buffer_tokens = item_tokens
        else:
            buffer.append(item)
            buffer_tokens += item_tokens

    if buffer:
        groups.append("\n".join(buffer))

    return groups


def recursive_split(
    text: str, separators: list[str], max_tokens: int, overlap_tokens: int
) -> list[str]:
    """
    Recursively split text on the first separator that produces pieces
    small enough to fit max_tokens, falling back to the next separator
    (down to character-level splitting) when a piece is still too large.
    Adjacent chunks overlap by overlap_tokens to preserve context.
    """
    pieces = _split_on_first_fitting_separator(text, separators, max_tokens)
    return _merge_pieces_with_overlap(pieces, max_tokens, overlap_tokens)


def _split_on_first_fitting_separator(
    text: str, separators: list[str], max_tokens: int
) -> list[str]:
    """Break text into the smallest set of pieces using the separator hierarchy."""
    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        return [text]  # character-level fallback already exhausted

    separator, *remaining_separators = separators
    pieces = text.split(separator) if separator else list(text)

    result: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if count_tokens(piece) > max_tokens:
            result.extend(
                _split_on_first_fitting_separator(piece, remaining_separators, max_tokens)
            )
        else:
            result.append(piece)
    return result


def _merge_pieces_with_overlap(
    pieces: list[str], max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Greedily pack small pieces into chunks near max_tokens, carrying overlap forward."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if buffer_tokens + piece_tokens > max_tokens and buffer:
            chunk_text = " ".join(buffer)
            chunks.append(chunk_text)

            # Carry the tail of this chunk forward as overlap context.
            overlap_text = _tail_by_tokens(chunk_text, overlap_tokens)
            buffer = [overlap_text] if overlap_text else []
            buffer_tokens = count_tokens(overlap_text) if overlap_text else 0

        buffer.append(piece)
        buffer_tokens += piece_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def _tail_by_tokens(text: str, token_count: int) -> str:
    """Return the trailing slice of text containing roughly token_count tokens."""
    if token_count <= 0:
        return ""
    tokens = _encoder.encode(text)
    return _encoder.decode(tokens[-token_count:])


def build_chunk(
    text: str,
    element_type: str,
    chunk_index: int,
    element: dict[str, Any],
    doc_name: str,
    doc_hash: str,
) -> dict[str, Any]:
    """Assemble the final chunk dict stored alongside its Qdrant point."""
    return {
        "chunk_id": f"{doc_name}_chunk_{chunk_index}",
        "text": text,
        "element_type": element_type,
        "section": element["section"],
        "page_number": element["page_number"],
        "doc_name": doc_name,
        "chunk_index": chunk_index,
        "token_count": count_tokens(text),
        "doc_hash": doc_hash,
    }
