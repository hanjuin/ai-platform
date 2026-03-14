import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

def get_cache(key: str):
    value = redis_client.get(key)
    if value:
        return json.loads(value)
    return None

def set_cache(key: str, value, ttl: int = 300):
    redis_client.setex(
        key,
        ttl,
        json.dumps(value)
    )