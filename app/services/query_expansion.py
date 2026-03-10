import hashlib
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from app.services.cache_service import get_cache, set_cache

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXPANSION_TTL = 3600  # 1 hour — query variants are stable


def expand_query(query: str, n: int = 4) -> list[str]:
    """
    Generate n alternative phrasings of the query using GPT-4o-mini.
    Results are cached in Redis by query hash.
    Returns the original query plus variants.
    """
    cache_key = f"qexp:{hashlib.sha256(query.encode()).hexdigest()}:{n}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate alternative phrasings of a search query to improve document retrieval. "
                    "Output ONLY a JSON array of strings — no explanation, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate {n} alternative phrasings of this query. "
                    f"Vary vocabulary and structure but preserve meaning.\n\nQuery: {query}"
                ),
            },
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    try:
        variants = json.loads(raw)
        if not isinstance(variants, list):
            variants = []
    except json.JSONDecodeError:
        variants = []

    results = [query] + [v for v in variants if isinstance(v, str) and v != query]
    set_cache(cache_key, results, ttl=EXPANSION_TTL)
    return results
