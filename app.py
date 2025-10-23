# app.py
import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from fastapi import FastAPI, Query, Header, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# DB & redis
import mysql.connector
from mysql.connector import pooling
import redis

# concurrency helpers
from starlette.concurrency import run_in_threadpool

# OpenAI
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, Timeout as APITimeoutError

# ---------------------------------------
# Load config & logging
# ---------------------------------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS")  # REQUIRED in env ideally
DB_NAME = os.getenv("DB_NAME", "evenza")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))

REDIS_URL = os.getenv("REDIS_URL", "")
API_SECRET = os.getenv("API_SECRET", "")  # optional header auth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if DB_PASS is None:
    logging.warning("DB_PASS not set in env — don't use hard-coded credentials in production!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evenza-backend")

# ---------------------------------------
# MySQL connection pooling
# ---------------------------------------
try:
    pool = pooling.MySQLConnectionPool(
        pool_name="evenza_pool",
        pool_size=DB_POOL_SIZE,
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        pool_reset_session=True,
    )
    logger.info("MySQL pool created (size=%s)", DB_POOL_SIZE)
except Exception as e:
    logger.exception("Failed to create MySQL connection pool: %s", e)
    raise

# ---------------------------------------
# Redis client (optional caching)
# ---------------------------------------
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _ = redis_client.ping()
        logger.info("Connected to Redis at %s", REDIS_URL)
    except Exception as e:
        logger.exception("Failed to connect to Redis, continuing without cache: %s", e)
        redis_client = None
else:
    logger.info("No REDIS_URL provided — cache disabled")

# ---------------------------------------
# OpenAI client (optional)
# ---------------------------------------
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
else:
    logger.info("OPENAI_API_KEY not set — chat endpoint will be limited")

# ---------------------------------------
# FastAPI app + optional API key dependency
# ---------------------------------------
app = FastAPI(title="Evenza Events API (production-ready)")

# CORS for local development (adjust origins for your frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit OPTIONS handler to satisfy preflight checks for chat endpoint
@app.options("/chat")
async def options_chat():
    return Response(status_code=204)

def require_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple optional header auth — set API_SECRET in .env to enforce."""
    if API_SECRET:
        if not x_api_key or x_api_key != API_SECRET:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ---------------------------------------
# Pydantic models (response schemas)
# ---------------------------------------
class EventOut(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    location: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    image: Optional[str] = None
    registered_at: Optional[str] = None
    saved_at: Optional[str] = None
    reg_count: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None

# ---------------------------------------
# DB helper utilities (sync) - run in threadpool
# ---------------------------------------
def _get_conn():
    """Return a pooled connection. Caller must close()."""
    return pool.get_connection()

def _fetchall_dict(sql: str, params: Tuple = ()):
    conn = _get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        return rows
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def _fetchone_dict(sql: str, params: Tuple = ()):
    conn = _get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

# ---------------------------------------
# Normalize DB rows -> EventOut dict
# ---------------------------------------
def _map_event(row: Dict) -> Dict:
    if row is None:
        return {}
    mapped = {
        "id": int(row.get("event_id")),
        "title": row.get("title"),
        "description": row.get("description"),
        "category_id": row.get("category_id"),
        "category_name": row.get("category_name"),
        "location": row.get("location"),
        "start_time": str(row.get("start_time")) if row.get("start_time") else None,
        "end_time": str(row.get("end_time")) if row.get("end_time") else None,
        "image": row.get("image"),
    }
    if "registered_at" in row:
        mapped["registered_at"] = str(row.get("registered_at")) if row.get("registered_at") else None
    if "saved_at" in row:
        mapped["saved_at"] = str(row.get("saved_at")) if row.get("saved_at") else None
    if "reg_count" in row:
        mapped["reg_count"] = int(row.get("reg_count") or 0)
    return mapped

# ---------------------------------------
# Caching helpers (Redis)
# ---------------------------------------
def cache_get(key: str):
    if not redis_client:
        return None
    try:
        val = redis_client.get(key)
        if val:
            return json.loads(val)
    except Exception as e:
        logger.debug("Redis get error for %s: %s", key, e)
    return None

def cache_set(key: str, value, ttl_seconds: int = 60):
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:
        logger.debug("Redis set error for %s: %s", key, e)

# ---------------------------------------
# Core fetch functions (sync)
# ---------------------------------------
def fetch_upcoming_events_sync(limit: int = 20, offset: int = 0) -> List[Dict]:
    key = f"upcoming:{limit}:{offset}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    sql = """
        SELECT e.event_id, e.title, e.description, e.category_id, c.name AS category_name,
               e.location, e.start_time, e.end_time, e.image
        FROM events e
        LEFT JOIN categories c ON c.category_id = e.category_id
        WHERE e.start_time >= NOW()
        ORDER BY e.start_time ASC
        LIMIT %s OFFSET %s
    """
    rows = _fetchall_dict(sql, (limit, offset))
    mapped = [_map_event(r) for r in rows]
    cache_set(key, mapped, ttl_seconds=30)
    return mapped

def fetch_registered_events_sync(user_id: int, only_future: bool = False, only_completed: bool = False, limit: int = 100, offset: int = 0) -> List[Dict]:
    sql = """
        SELECT e.event_id, e.title, e.description, e.category_id, c.name AS category_name,
               e.location, e.start_time, e.end_time, e.image, r.registered_at
        FROM registrations r
        JOIN events e ON r.event_id = e.event_id
        LEFT JOIN categories c ON c.category_id = e.category_id
        WHERE r.user_id = %s
    """
    params = [user_id]
    if only_future:
        sql += " AND e.start_time >= NOW()"
    elif only_completed:
        sql += " AND e.end_time < NOW()"
    sql += " ORDER BY e.start_time DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    rows = _fetchall_dict(sql, tuple(params))
    mapped = []
    for r in rows:
        ev = _map_event(r)
        ev["registered_at"] = str(r.get("registered_at")) if r.get("registered_at") else None
        mapped.append(ev)
    return mapped

def fetch_saved_events_sync(user_id: int, only_future: bool = False, limit: int = 100, offset: int = 0) -> List[Dict]:
    sql = """
        SELECT e.event_id, e.title, e.description, e.category_id, c.name AS category_name,
               e.location, e.start_time, e.end_time, e.image, se.created_at AS saved_at
        FROM saved_events se
        JOIN events e ON se.event_id = e.event_id
        LEFT JOIN categories c ON c.category_id = e.category_id
        WHERE se.user_id = %s
    """
    params = [user_id]
    if only_future:
        sql += " AND e.start_time >= NOW()"
    sql += " ORDER BY e.start_time DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    rows = _fetchall_dict(sql, tuple(params))
    mapped = []
    for r in rows:
        ev = _map_event(r)
        ev["saved_at"] = str(r.get("saved_at")) if r.get("saved_at") else None
        mapped.append(ev)
    return mapped

def fetch_trending_events_sync(limit: int = 20, offset: int = 0) -> List[Dict]:
    key = f"trending:{limit}:{offset}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    sql = """
        SELECT e.event_id, e.title, e.description, e.category_id, c.name AS category_name,
               e.location, e.start_time, e.end_time, e.image, COUNT(r.registration_id) AS reg_count
        FROM events e
        LEFT JOIN registrations r ON r.event_id = e.event_id
        LEFT JOIN categories c ON c.category_id = e.category_id
        WHERE e.start_time >= NOW()
        GROUP BY e.event_id
        ORDER BY reg_count DESC, e.start_time ASC
        LIMIT %s OFFSET %s
    """
    rows = _fetchall_dict(sql, (limit, offset))
    mapped = [_map_event(r) for r in rows]
    cache_set(key, mapped, ttl_seconds=60)
    return mapped

def _aggregate_user_category_scores_sync(user_id: int) -> List[Tuple[int, int]]:
    def _fetch_counts(sql: str, params: Tuple) -> Dict[int, int]:
        rows = _fetchall_dict(sql, params)
        return {r["category_id"]: int(r["cnt"]) for r in rows if r.get("category_id") is not None}

    regs = _fetch_counts(
        """
        SELECT e.category_id, COUNT(*) AS cnt
        FROM registrations r
        JOIN events e ON e.event_id = r.event_id
        WHERE r.user_id = %s
        GROUP BY e.category_id
        """,
        (user_id,)
    )
    saved = _fetch_counts(
        """
        SELECT e.category_id, COUNT(*) AS cnt
        FROM saved_events s
        JOIN events e ON e.event_id = s.event_id
        WHERE s.user_id = %s
        GROUP BY e.category_id
        """,
        (user_id,)
    )
    reviews = _fetch_counts(
        """
        SELECT e.category_id, COUNT(*) AS cnt
        FROM ratings_reviews rr
        JOIN events e ON e.event_id = rr.event_id
        WHERE rr.user_id = %s
        GROUP BY e.category_id
        """,
        (user_id,)
    )
    cats: Dict[int, int] = {}
    for cid, cnt in regs.items():
        cats[cid] = cats.get(cid, 0) + cnt * 3
    for cid, cnt in saved.items():
        cats[cid] = cats.get(cid, 0) + cnt * 2
    for cid, cnt in reviews.items():
        cats[cid] = cats.get(cid, 0) + cnt * 1
    ordered = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
    return ordered

def fetch_recommendations_sync(user_id: int, limit: int = 20, top_n: int = 3) -> List[Dict]:
    ordered = _aggregate_user_category_scores_sync(user_id)
    top_categories = [cid for cid, score in ordered[:top_n] if score > 0]

    params: List = []
    where_cat = ""
    if top_categories:
        placeholders = ",".join(["%s"] * len(top_categories))
        where_cat = f" AND e.category_id IN ({placeholders})"
        params.extend(top_categories)

    params.extend([user_id, user_id, limit])
    sql = (
        """
        SELECT e.event_id, e.title, e.description, e.category_id, c.name AS category_name,
               e.location, e.start_time, e.end_time, e.image
        FROM events e
        LEFT JOIN categories c ON c.category_id = e.category_id
        WHERE e.start_time >= NOW()
        """
        + where_cat +
        """
        AND e.event_id NOT IN (SELECT event_id FROM registrations WHERE user_id = %s)
        AND e.event_id NOT IN (SELECT event_id FROM saved_events WHERE user_id = %s)
        ORDER BY e.start_time ASC
        LIMIT %s
        """
    )
    rows = _fetchall_dict(sql, tuple(params))
    events = [_map_event(r) for r in rows]
    if len(events) < limit:
        more = fetch_upcoming_events_sync(limit)
        seen = {e["id"] for e in events}
        for ev in more:
            if ev["id"] not in seen:
                events.append(ev)
            if len(events) >= limit:
                break
    return events

def fetch_mix_feed_sync(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
    registered_all = fetch_registered_events_sync(user_id, only_future=False, only_completed=False, limit=500, offset=0)
    saved_all = fetch_saved_events_sync(user_id, only_future=False, limit=500, offset=0)
    upcoming = fetch_upcoming_events_sync(limit=limit, offset=offset)
    trending = fetch_trending_events_sync(limit=limit, offset=offset)
    combined = registered_all + saved_all + upcoming + trending
    feed: List[Dict] = []
    seen_ids = set()
    for ev in combined:
        ev_id = ev.get("id")
        if ev_id is None:
            continue
        if ev_id not in seen_ids:
            feed.append(ev)
            seen_ids.add(ev_id)
            if len(feed) >= limit:
                break
    return feed

# ---------------------------------------
# Async wrappers calling blocking sync code in a threadpool
# ---------------------------------------
async def fetch_upcoming_events(limit: int = 20, offset: int = 0):
    return await run_in_threadpool(fetch_upcoming_events_sync, limit, offset)

async def fetch_registered_events(user_id: int, only_future: bool = False, only_completed: bool = False, limit: int = 100, offset: int = 0):
    return await run_in_threadpool(fetch_registered_events_sync, user_id, only_future, only_completed, limit, offset)

async def fetch_saved_events(user_id: int, only_future: bool = False, limit: int = 100, offset: int = 0):
    return await run_in_threadpool(fetch_saved_events_sync, user_id, only_future, limit, offset)

async def fetch_trending_events(limit: int = 20, offset: int = 0):
    return await run_in_threadpool(fetch_trending_events_sync, limit, offset)

async def fetch_recommendations(user_id: int, limit: int = 20, top_n: int = 3):
    return await run_in_threadpool(fetch_recommendations_sync, user_id, limit, top_n)

async def fetch_mix_feed(user_id: int, limit: int = 50, offset: int = 0):
    return await run_in_threadpool(fetch_mix_feed_sync, user_id, limit, offset)

# ---------------------------------------
# API Endpoints
# ---------------------------------------
@app.get("/events/upcoming", response_model=List[EventOut])
async def api_upcoming(limit: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0), _auth=Depends(require_api_key)):
    rows = await fetch_upcoming_events(limit=limit, offset=offset)
    return [EventOut(**r) for r in rows]

@app.get("/events/registered/{user_id}", response_model=List[EventOut])
async def api_registered(
    user_id: int,
    future: bool = Query(False),
    completed: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_api_key),
):
    if future and completed:
        completed = False
    rows = await fetch_registered_events(user_id, only_future=future, only_completed=completed, limit=limit, offset=offset)
    return [EventOut(**r) for r in rows]

@app.get("/events/saved/{user_id}", response_model=List[EventOut])
async def api_saved(user_id: int, future: bool = Query(False), limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), _auth=Depends(require_api_key)):
    rows = await fetch_saved_events(user_id, only_future=future, limit=limit, offset=offset)
    return [EventOut(**r) for r in rows]

@app.get("/events/trending", response_model=List[EventOut])
async def api_trending(limit: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0), _auth=Depends(require_api_key)):
    rows = await fetch_trending_events(limit=limit, offset=offset)
    return [EventOut(**r) for r in rows]

@app.get("/events/recommendations/{user_id}", response_model=List[EventOut])
async def api_recommendations(user_id: int, limit: int = Query(20, ge=1, le=500), top_n: int = Query(3, ge=1, le=10), _auth=Depends(require_api_key)):
    rows = await fetch_recommendations(user_id, limit=limit, top_n=top_n)
    return [EventOut(**r) for r in rows]

@app.get("/events/mix/{user_id}", response_model=List[EventOut])
async def api_mix(user_id: int, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), _auth=Depends(require_api_key)):
    rows = await fetch_mix_feed(user_id, limit=limit, offset=offset)
    return [EventOut(**r) for r in rows]

@app.get("/events/feeds/{user_id}")
async def api_all_feeds(user_id: int, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), _auth=Depends(require_api_key)):
    return {
        "registered_events": await fetch_registered_events(user_id, limit=limit, offset=offset),
        "upcoming_events": await fetch_upcoming_events(limit=limit, offset=offset),
        "mix_feed": await fetch_mix_feed(user_id, limit=limit, offset=offset),
        "recommendations": await fetch_recommendations(user_id, limit=limit),
        "trending": await fetch_trending_events(limit=limit, offset=offset),
    }

# ---------------------------------------
# Improved Chat endpoint with tools and caching
# ---------------------------------------
# Helper to format events as text for the assistant
def _events_to_text(events: list) -> str:
    if not events:
        return "No events found."
    lines = []
    for e in events:
        title = e.get("title") or e.get("name") or "Untitled event"
        st = e.get("start_time") or ""
        loc = e.get("location") or ""
        cat = e.get("category_name") or ""
        lines.append(f"- {title} | {cat} | {st} | {loc}")
    return "\n".join(lines)

# caching wrapper for sync tool functions
def _cached_tool_result(key: str, fn, ttl: int = 20, *args, **kwargs):
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
    res = fn(*args, **kwargs)
    if redis_client:
        try:
            redis_client.setex(key, ttl, json.dumps(res, default=str))
        except Exception:
            pass
    return res

# robust OpenAI call with retries/backoff
def _openai_chat_create_with_retries(payload: Dict[str, Any], max_attempts: int = 3, backoff_base: float = 0.8):
    attempt = 0
    last_exc = None
    while attempt < max_attempts:
        try:
            return openai_client.chat.completions.create(**payload)
        except Exception as e:
            last_exc = e
            attempt += 1
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning("OpenAI call failed (attempt %s/%s): %s — retrying in %.2fs", attempt, max_attempts, str(e), wait)
            time.sleep(wait)
    logger.exception("OpenAI calls exhausted: %s", last_exc)
    raise last_exc

# Tool handlers (sync functions)
def _handle_get_upcoming(args: dict):
    limit = int(args.get("limit", 10))
    offset = int(args.get("offset", 0))
    key = f"tool:upcoming:{limit}:{offset}"
    return _cached_tool_result(key, fetch_upcoming_events_sync, ttl=20, limit=limit, offset=offset)

def _handle_get_registered(args: dict, user_id_hint: int | None):
    user_id = int(args.get("user_id") or user_id_hint or 0)
    future = bool(args.get("future", False))
    completed = bool(args.get("completed", False))
    limit = int(args.get("limit", 100))
    offset = int(args.get("offset", 0))
    key = f"tool:registered:{user_id}:{future}:{completed}:{limit}:{offset}"
    return _cached_tool_result(key, fetch_registered_events_sync, ttl=20, user_id=user_id, only_future=future, only_completed=completed, limit=limit, offset=offset)

def _handle_get_saved(args: dict, user_id_hint: int | None):
    user_id = int(args.get("user_id") or user_id_hint or 0)
    future = bool(args.get("future", False))
    limit = int(args.get("limit", 100))
    offset = int(args.get("offset", 0))
    key = f"tool:saved:{user_id}:{future}:{limit}:{offset}"
    return _cached_tool_result(key, fetch_saved_events_sync, ttl=20, user_id=user_id, only_future=future, limit=limit, offset=offset)

def _handle_get_mix(args: dict, user_id_hint: int | None):
    user_id = int(args.get("user_id") or user_id_hint or 0)
    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))
    key = f"tool:mix:{user_id}:{limit}:{offset}"
    return _cached_tool_result(key, fetch_mix_feed_sync, ttl=20, user_id=user_id, limit=limit, offset=offset)

def _handle_get_recs(args: dict, user_id_hint: int | None):
    user_id = int(args.get("user_id") or user_id_hint or 0)
    limit = int(args.get("limit", 20))
    top_n = int(args.get("top_n", 3))
    key = f"tool:recs:{user_id}:{limit}:{top_n}"
    return _cached_tool_result(key, fetch_recommendations_sync, ttl=30, user_id=user_id, limit=limit, top_n=top_n)

def _handle_get_trending(args: dict):
    limit = int(args.get("limit", 10))
    offset = int(args.get("offset", 0))
    key = f"tool:trending:{limit}:{offset}"
    return _cached_tool_result(key, fetch_trending_events_sync, ttl=30, limit=limit, offset=offset)

_TOOL_HANDLERS = {
    "get_upcoming_events": lambda args, user_hint=None: _handle_get_upcoming(args),
    "get_registered_events": lambda args, user_hint=None: _handle_get_registered(args, user_hint),
    "get_saved_events": lambda args, user_hint=None: _handle_get_saved(args, user_hint),
    "get_mix_feed": lambda args, user_hint=None: _handle_get_mix(args, user_hint),
    "get_recommendations": lambda args, user_hint=None: _handle_get_recs(args, user_hint),
    "get_trending": lambda args, user_hint=None: _handle_get_trending(args),
}

@app.post("/chat")
async def chat_with_ai_improved(request: ChatRequest, _auth=Depends(require_api_key)):
    """
    Enhanced chat endpoint:
      - Exposes multiple tools for the model
      - Uses caching and retries
      - Returns both assistant reply and structured events
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured (OPENAI_API_KEY missing)")

    tools = [
        {"type": "function", "function": {"name": "get_upcoming_events", "description": "Get upcoming public events. Args: {limit,offset}", "parameters": {"type": "object","properties": {"limit": {"type":"integer"},"offset":{"type":"integer"}}}}},
        {"type": "function", "function": {"name": "get_registered_events", "description": "Get events the user registered for. Args: {user_id, future, completed, limit, offset}", "parameters": {"type": "object","properties": {"user_id":{"type":"integer"},"future":{"type":"boolean"},"completed":{"type":"boolean"},"limit":{"type":"integer"},"offset":{"type":"integer"}}}}},
        {"type": "function", "function": {"name": "get_saved_events", "description": "Get events the user saved/bookmarked. Args: {user_id, future, limit, offset}", "parameters": {"type": "object","properties": {"user_id":{"type":"integer"},"future":{"type":"boolean"},"limit":{"type":"integer"},"offset":{"type":"integer"}}}}},
        {"type": "function", "function": {"name": "get_mix_feed", "description": "Get mixed feed for a user. Args: {user_id, limit, offset}", "parameters": {"type": "object","properties": {"user_id":{"type":"integer"},"limit":{"type":"integer"},"offset":{"type":"integer"}}}}},
        {"type": "function", "function": {"name": "get_recommendations", "description": "Recommend upcoming events for a user. Args: {user_id, limit, top_n}", "parameters": {"type": "object","properties": {"user_id":{"type":"integer"},"limit":{"type":"integer"},"top_n":{"type":"integer"}}}}},
        {"type": "function", "function": {"name": "get_trending", "description": "Get currently trending upcoming events. Args: {limit, offset}", "parameters": {"type":"object","properties": {"limit":{"type":"integer"},"offset":{"type":"integer"}}}}},
    ]

    system_prompt = (
        "You are Evenza AI Assistant. When the user asks for events or event info, prefer to call the available tools "
        "to fetch exact event data. Use tools when you need fresh or personal data. Return concise and actionable replies. "
        "If you call a tool, wait for its results and then produce a single natural-language reply summarizing the results. "
        "If multiple tools are useful, call at most one tool per assistant turn and ask a follow-up question only if needed."
    )

    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        response = await run_in_threadpool(_openai_chat_create_with_retries, payload)

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            tc = tool_calls[0]
            # Support both SDK object attributes and dict-like access
            if hasattr(tc, "function"):
                fn_name = getattr(tc.function, "name", None)
                args_raw = getattr(tc.function, "arguments", "{}")
                tool_id = getattr(tc, "id", None)
            else:
                fn_name = tc.get("function", {}).get("name")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                tool_id = tc.get("id")

            # Keep original string for the assistant tool call; also parse to dict for handler
            args_str = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
            try:
                args = json.loads(args_str) if args_str else {}
            except Exception:
                args = {}

            if fn_name not in _TOOL_HANDLERS:
                tool_output = {"error": f"Tool {fn_name} not supported."}
            else:
                handler = _TOOL_HANDLERS[fn_name]
                tool_res = await run_in_threadpool(handler, args, request.user_id)
                tool_output = tool_res

            events_text = _events_to_text(tool_output if isinstance(tool_output, list) else [])
            assistant_with_tool_call = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str
                        }
                    }
                ],
            }

            followup_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are Evenza AI Assistant."},
                    {"role": "user", "content": request.message},
                    assistant_with_tool_call,
                    {"role": "tool", "tool_call_id": tool_id, "content": events_text},
                ],
                "temperature": 0.2,
            }

            followup_resp = await run_in_threadpool(_openai_chat_create_with_retries, followup_payload)
            content = getattr(followup_resp.choices[0].message, "content", "")
            return {"reply": content, "ai_text": content}

        content = getattr(message, "content", "")
        return {"reply": content, "ai_text": content}

    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="OpenAI request timed out")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="OpenAI authentication error")
    except APIError as e:
        logger.exception("OpenAI API error: %s", e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.exception("Unexpected chat error: %s", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# ---------------------------------------
# Health & simple admin endpoints
# ---------------------------------------
@app.get("/health")
async def health():
    healthy = {"db": False, "redis": False}
    try:
        conn = _get_conn()
        conn.close()
        healthy["db"] = True
    except Exception:
        healthy["db"] = False
    if redis_client:
        try:
            healthy["redis"] = redis_client.ping()
        except Exception:
            healthy["redis"] = False
    else:
        healthy["redis"] = None
    return {"ok": True, "services": healthy}

# ---------------------------------------
# Notes
# ---------------------------------------
# - Ensure environment variables are set (DB_PASS, OPENAI_API_KEY if using chat)
# - Recommended (manual) DB index checks:
#   SHOW INDEX FROM events;
#   EXPLAIN SELECT ...;
# - For production, run behind an API gateway, TLS, and a secrets manager.
