from markdown_it import MarkdownIt
import tiktoken
from typing import List, Tuple
from app.modules.chunk_content.semantic_chunk import semantic_chunk


MAX_TOKENS = 200


def parse_markdown_sections(md_text: str) -> List[Tuple[List[str], str]]:
    md = MarkdownIt()
    tokens = md.parse(md_text)

    sections: List[Tuple[List[str], str]] = []
    stack: List[str] = []
    current_text = ""

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.type == "heading_open":
            level = int(tok.tag[1])

            if current_text.strip():
                sections.append((stack.copy(), current_text.strip()))
                current_text = ""

            inline_token = tokens[i + 1]
            heading_text = inline_token.content if inline_token.type == "inline" else ""

            stack = stack[: level - 1]
            stack.append(heading_text)

            i += 3
            continue

        # Capture all inline content (not just paragraph)
        if tok.type == "inline":
            current_text += tok.content + "\n"

        i += 1

    if current_text.strip():
        sections.append((stack.copy(), current_text.strip()))

    return sections


def breadcrumb(section_path: List[str]) -> str:
    return f"[PATH: {' > '.join(section_path)}]"


def chunk_doc_by_headings(text: str) -> List[str]:

    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = text.strip()

    if not text: 
        return []

    chunks: List[str] = []

    sections = parse_markdown_sections(text)

    for section_path, section_text in sections:

        if not section_text.strip():
            continue

        token_count = len(tokenizer.encode(section_text))
        path_prefix = breadcrumb(section_path)

        # Small section → keep as one chunk
        if token_count <= MAX_TOKENS:
            chunks.append(f"{path_prefix}\n\n{section_text.strip()}")
            continue

        # Large section → semantic split
        subchunks = semantic_chunk(section_text)

        for subchunk in subchunks:
            if not isinstance(subchunk, str):
                raise TypeError(
                    f"semantic_chunk must return str chunks, got {type(subchunk)}"
                )

            chunks.append(f"{path_prefix}\n\n{subchunk.strip()}")

    return chunks