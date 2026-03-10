from dataclasses import dataclass, field
from markdown_it import MarkdownIt
import tiktoken
from typing import List, Tuple
from app.modules.chunk_content.semantic_chunk import semantic_chunk


CHILD_MAX_TOKENS = 200   # max tokens per child chunk (used for retrieval)


@dataclass
class ChunkGroup:
    """
    One markdown section = one ChunkGroup.
    - header: breadcrumb path (e.g. "Intro > Background"), stored separately, not embedded
    - parent_content: full section text, stored as a parent chunk (no embedding)
    - children: list of subchunk texts to embed for retrieval
    """
    header: str
    parent_content: str
    children: List[str] = field(default_factory=list)


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

        if tok.type == "inline":
            current_text += tok.content + "\n"

        i += 1

    if current_text.strip():
        sections.append((stack.copy(), current_text.strip()))

    return sections


def breadcrumb(section_path: List[str]) -> str:
    return " > ".join(section_path)


def chunk_doc_by_headings(text: str) -> List[ChunkGroup]:
    """
    Returns a list of ChunkGroup, one per markdown section.

    Each group has:
    - header: breadcrumb (not embedded)
    - parent_content: full section text (stored as parent chunk, no embedding)
    - children: subchunk texts to embed for retrieval

    The parent-child design means retrieval uses small, precise child chunks,
    but the LLM receives the full parent section for richer context.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = text.strip()

    if not text:
        return []

    groups: List[ChunkGroup] = []
    sections = parse_markdown_sections(text)

    for section_path, section_text in sections:
        if not section_text.strip():
            continue

        header = breadcrumb(section_path)
        token_count = len(tokenizer.encode(section_text))

        if token_count <= CHILD_MAX_TOKENS:
            # Small section: parent and child are the same text
            groups.append(ChunkGroup(
                header=header,
                parent_content=section_text.strip(),
                children=[section_text.strip()],
            ))
        else:
            # Large section: semantic split into children; parent holds full text
            subchunks = semantic_chunk(section_text)
            valid_children = [
                s.strip() for s in subchunks
                if isinstance(s, str) and s.strip()
            ]
            groups.append(ChunkGroup(
                header=header,
                parent_content=section_text.strip(),
                children=valid_children if valid_children else [section_text.strip()],
            ))

    return groups
