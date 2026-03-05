import re
import numpy as np
import tiktoken
from typing import List
from app.services.embedding_service import embedding_service

SIMILARITY_THRESHOLD = 0.6
MAX_TOKENS = 200
MIN_TOKENS = 30
OVERLAP_TOKENS = 20
USE_OVERLAP = True

def _cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Fast cosine similarity without sklearn."""
    return float(
        np.dot(vec_a, vec_b) /
        (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)
    )

def semantic_chunk(text: str) -> List[str]:
    """
    Perform semantic + token-aware chunking.

    Steps:
    1. Split into sentences
    2. Embed sentences
    3. Merge sentences based on similarity + token constraints
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        return []

    # Improved regex: keep last sentence even without punctuation
    sentences = re.findall(r"[^.?!]+[.?!]?", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    if len(sentences) == 1:
        return [sentences[0]]

    # Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Embed all sentences (batch)
    embeddings = embedding_service.generate_embedding(sentences)

    if len(embeddings) != len(sentences):
        raise ValueError("Embedding count mismatch with sentences")

    embeddings = np.asarray(embeddings)

    chunks: List[str] = []

    current_chunk = sentences[0]
    current_tokens = tokenizer.encode(current_chunk)
    current_token_len = len(current_tokens)

    for i in range(1, len(sentences)):

        vec_prev = embeddings[i - 1]
        vec_curr = embeddings[i]

        sim = _cosine_sim(vec_prev, vec_curr)

        next_sentence = sentences[i]
        next_tokens = tokenizer.encode(next_sentence)
        next_token_len = len(next_tokens)

        semantic_break = sim <= SIMILARITY_THRESHOLD
        token_overflow = current_token_len + next_token_len > MAX_TOKENS
        enough_tokens = current_token_len >= MIN_TOKENS

        should_break = token_overflow or (semantic_break and enough_tokens)

        if should_break:
            # Save finished chunk
            chunks.append(current_chunk.strip())

            # Handle overlap if breaking due to token overflow
            if USE_OVERLAP and token_overflow and not semantic_break:
                overlap_slice = current_tokens[
                    -min(OVERLAP_TOKENS, len(current_tokens)) :
                ]
                overlap_text = tokenizer.decode(overlap_slice).strip()

                current_chunk = overlap_text + " " + next_sentence
                current_tokens = tokenizer.encode(current_chunk)
                current_token_len = len(current_tokens)
            else:
                current_chunk = next_sentence
                current_tokens = next_tokens
                current_token_len = next_token_len

        else:
            current_chunk += " " + next_sentence
            current_tokens.extend(next_tokens)
            current_token_len += next_token_len

    # Final chunk
    chunks.append(current_chunk.strip())

    return chunks