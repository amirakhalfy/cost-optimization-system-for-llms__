import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from cachetools import TTLCache
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.db_setup import get_db
from app.db.models import PromptCache

# --------- LOGGING SETUP ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- CONFIG ---------
class CacheConfig:
    # Redis Configuration
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6380
    REDIS_DB = 0
    REDIS_PASSWORD = None
    REDIS_SOCKET_TIMEOUT = 5
    REDIS_SOCKET_CONNECT_TIMEOUT = 5
    REDIS_RETRY_ON_TIMEOUT = True
    REDIS_HEALTH_CHECK_INTERVAL = 30
    
    # Cache TTL Configuration
    CACHE_TTL_SECONDS = 3600  # 1 hour
    LOCAL_CACHE_TTL = 300     # 5 minutes
    LONG_TERM_CACHE_TTL = 86400 * 7  # 7 days
    
    # Local Cache Configuration
    LOCAL_CACHE_SIZE = 512    # Reduced for production
    
    # Performance Configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 0.1
    CIRCUIT_BREAKER_THRESHOLD = 10
    CIRCUIT_BREAKER_TIMEOUT = 60

config = CacheConfig()

# --------- METRICS & MONITORING ---------
class CacheMetrics:
    def __init__(self):
        """
        Initialize cache metrics.
        """
        self.hits = {"local": 0, "redis": 0, "database": 0}
        self.misses = {"local": 0, "redis": 0, "database": 0}
        self.errors = {"redis": 0, "database": 0, "llm": 0}
        self.response_times = {"local": [], "redis": [], "database": [], "llm": []}
        self.circuit_breaker_trips = 0
        
    def record_hit(self, cache_type: str, response_time: float = None):
        """"Record a cache hit with optional response time."""
        self.hits[cache_type] = self.hits.get(cache_type, 0) + 1
        if response_time and cache_type in self.response_times:
            self.response_times[cache_type].append(response_time)
            if len(self.response_times[cache_type]) > 1000:
                self.response_times[cache_type] = self.response_times[cache_type][-1000:]
    
    def record_miss(self, cache_type: str):
        """"Record a cache miss."""
        self.misses[cache_type] = self.misses.get(cache_type, 0) + 1
    
    def record_error(self, error_type: str):
        """"Record an error for a specific cache type."""
        self.errors[error_type] = self.errors.get(error_type, 0) + 1
    
    def get_hit_rate(self, cache_type: str) -> float:
        """Calculate hit rate for a specific cache type."""
        hits = self.hits.get(cache_type, 0)
        misses = self.misses.get(cache_type, 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def get_avg_response_time(self, cache_type: str) -> float:
        """Calculate average response time for a specific cache type."""
        times = self.response_times.get(cache_type, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rates": {k: self.get_hit_rate(k) for k in self.hits.keys()},
            "avg_response_times": {k: self.get_avg_response_time(k) for k in self.response_times.keys()},
            "circuit_breaker_trips": self.circuit_breaker_trips
        }

metrics = CacheMetrics()

# --------- CIRCUIT BREAKER ---------
class CircuitBreaker:
    """A simple circuit breaker implementation to prevent cascading failures in the cache system.
    """
    def __init__(self, failure_threshold: int = 10, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call a function with circuit breaker logic."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                metrics.circuit_breaker_trips += 1
            raise e

redis_circuit_breaker = CircuitBreaker(config.CIRCUIT_BREAKER_THRESHOLD, config.CIRCUIT_BREAKER_TIMEOUT)

# --------- REDIS CLIENT WITH FALLBACK ---------
class RedisManager:
    """Redis client manager with connection pooling and circuit breaker."""
    def __init__(self):
        self.redis_client = None
        self.connection_pool = None
        self.is_connected = False
        
    async def initialize(self):
        """Initialize Redis connection with error handling and circuit breaker."""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}",
                password=config.REDIS_PASSWORD,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
                retry_on_timeout=config.REDIS_RETRY_ON_TIMEOUT,
                health_check_interval=config.REDIS_HEALTH_CHECK_INTERVAL,
                max_connections=20,
            )
            self.redis_client = aioredis.Redis(connection_pool=self.connection_pool)
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.is_connected = False
            metrics.record_error("redis")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis with circuit breaker."""
        if not self.is_connected:
            return None
        
        try:
            return await redis_circuit_breaker.call(self.redis_client.get, key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            metrics.record_error("redis")
            return None
    
    async def setex(self, key: str, ttl: int, value: str) -> bool:
        """Set value in Redis with expiration using SETEX command."""
        if not self.is_connected:
            return False
        
        try:
            await redis_circuit_breaker.call(self.redis_client.setex, key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis SETEX error: {e}")
            metrics.record_error("redis")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.is_connected:
            return False
        
        try:
            await redis_circuit_breaker.call(self.redis_client.delete, key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            metrics.record_error("redis")
            return False
    
    async def cleanup(self):
        """Cleanup Redis connection pool on shutdown."""
        if self.connection_pool:
            await self.connection_pool.disconnect()

redis_manager = RedisManager()

# --------- LOCAL CACHE WITH LRU ---------
local_cache = TTLCache(maxsize=config.LOCAL_CACHE_SIZE, ttl=config.LOCAL_CACHE_TTL)

# --------- UTILITIES ---------
def generate_prompt_key(prompt: str, model_name: str, max_tokens: int, **kwargs) -> str:
    """Generate deterministic cache key from prompt parameters."""
    params = {
        "prompt": prompt,
        "model_name": model_name,
        "max_tokens": max_tokens,
        **{k: v for k, v in kwargs.items() if v is not None}
    }
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

async def simulate_llm_call(prompt: str, model_name: str, max_tokens: int, **kwargs) -> dict:
    """Simulate LLM API call with realistic delay."""
    start_time = time.time()
    
    try:
        await asyncio.sleep(0.5 + len(prompt) / 10000) 
        
        response_time = time.time() - start_time
        metrics.record_hit("llm", response_time)
        
        return {
            "response": f"Generated response for: {prompt[:50]}...",
            "input_tokens": len(prompt.split()),
            "output_tokens": min(max_tokens, 50),
            "model": model_name,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": min(max_tokens, 50),
                "total_tokens": len(prompt.split()) + min(max_tokens, 50)
            }
        }
    except Exception as e:
        metrics.record_error("llm")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

def serialize_cache_data(data: dict) -> bytes:
    """Serialize cache data using pickle for better performance."""
    return pickle.dumps(data)

def deserialize_cache_data(data: bytes) -> dict:
    """Deserialize cache data from pickle."""
    return pickle.loads(data)

# --------- SCHEMAS ---------
class PromptRequest(BaseModel):
    """Request schema for prompt caching."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    model_name: str = Field(..., min_length=1)
    max_tokens: int = Field(default=100, ge=1, le=4000)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    ttl_seconds: Optional[int] = Field(default=config.CACHE_TTL_SECONDS, ge=60, le=86400)
    user_email: Optional[str] = None
    force_refresh: bool = Field(default=False)

class PromptResponse(BaseModel):
    """Response schema for prompt caching."""
    cached: bool
    cache_type: Optional[str] = None
    cache_hit_level: Optional[str] = None  # L1, L2, L3, or None
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    response_time_ms: float
    finish_reason: str
    usage: dict

class CacheStats(BaseModel):
    """Comprehensive cache statistics and health metrics."""
    hits: dict
    misses: dict
    errors: dict
    hit_rates: dict
    avg_response_times: dict
    circuit_breaker_trips: int
    redis_connected: bool
    local_cache_size: int
    local_cache_maxsize: int

# --------- DATABASE OPERATIONS ---------
async def save_to_database(db: Session, key: str, req: PromptRequest, result: dict) -> bool:
    """Save cache entry to database for persistence and analytics."""
    try:
        # Check if the entry already exists
        existing = db.query(PromptCache).filter(PromptCache.prompt_key == key).first()
        
        if existing:
            # Update existing entry
            existing.response = result["response"]
            existing.input_tokens = result["input_tokens"]
            existing.output_tokens = result["output_tokens"]
            existing.timestamp = datetime.now()
            logger.debug(f"Updated existing cache entry in database: {key[:16]}...")
        else:
            # Create new entry
            db_cache = PromptCache(
                prompt_key=key,
                raw_prompt=req.prompt,
                model_name=req.model_name,
                max_tokens=req.max_tokens,
                response=result["response"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                timestamp=datetime.now()
            )
            db.add(db_cache)
            logger.debug(f"Saved new cache entry to database: {key[:16]}...")
        
        await asyncio.get_event_loop().run_in_executor(None, db.commit)
        return True
    except Exception as e:
        logger.error(f"Database save error: {e}")
        metrics.record_error("database")
        await asyncio.get_event_loop().run_in_executor(None, db.rollback)
        return False

async def get_from_database(db: Session, key: str) -> Optional[dict]:
    """Get cache entry from database (L3 cache)."""
    db_start_time = time.time()
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: db.query(PromptCache).filter(PromptCache.prompt_key == key).first()
        )
        
        if result:
            # Check if it's not too old (7 days max)
            if datetime.now() - result.timestamp < timedelta(seconds=config.LONG_TERM_CACHE_TTL):
                response_time = (time.time() - db_start_time) * 1000
                metrics.record_hit("database", response_time / 1000)
                logger.debug(f"Cache hit: Database (L3) - Key: {key[:16]}...")
                
                return {
                    "response": result.response,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "model": result.model_name,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": result.input_tokens,
                        "completion_tokens": result.output_tokens,
                        "total_tokens": result.input_tokens + result.output_tokens
                    }
                }
            else:
                # Entry is too old, delete it
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: (db.delete(result), db.commit())
                )
                logger.debug(f"Deleted expired cache entry from database: {key[:16]}...")
        
        metrics.record_miss("database")
        return None
    except Exception as e:
        logger.error(f"Database get error: {e}")
        metrics.record_error("database")
        return None

# --------- MULTI-LEVEL CACHE IMPLEMENTATION ---------
class MultiLevelCache:
    def __init__(self):
        self.local_cache = local_cache
        self.redis_manager = redis_manager
    
    async def get(self, key: str, db: Session = None) -> Tuple[Optional[dict], Optional[str]]:
        """Get from cache with multi-level fallback. Returns (data, cache_level)."""
        start_time = time.time()
        
        # Level 1: Local cache
        if key in self.local_cache:
            response_time = (time.time() - start_time) * 1000
            metrics.record_hit("local", response_time / 1000)
            logger.debug(f"Cache hit: Local (L1) - Key: {key[:16]}...")
            return self.local_cache[key], "L1"
        else:
            metrics.record_miss("local")
        
        # Level 2: Redis cache
        redis_start = time.time()
        cached_data = await self.redis_manager.get(key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                # Check TTL
                if "timestamp" in data:
                    cached_time = datetime.fromisoformat(data["timestamp"])
                    if datetime.now() - cached_time < timedelta(seconds=data.get("ttl", config.CACHE_TTL_SECONDS)):
                        # Store in local cache for next time
                        self.local_cache[key] = data
                        response_time = (time.time() - redis_start) * 1000
                        metrics.record_hit("redis", response_time / 1000)
                        logger.debug(f"Cache hit: Redis (L2) - Key: {key[:16]}...")
                        return data, "L2"
                    else:
                        # Expired, delete from redis
                        await self.redis_manager.delete(key  )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Invalid cache data format: {e}")
                await self.redis_manager.delete(key)
        
        metrics.record_miss("redis")
        
        # Level 3: Database cache
        if db:
            db_cached = await get_from_database(db, key)
            if db_cached:
                # Store back in L1 and L2 cache for next time
                await self.set(key, db_cached)
                response_time = (time.time() - start_time) * 1000
                logger.debug(f"Cache hit: Database (L3) promoted to L1/L2 - Key: {key[:16]}...")
                return db_cached, "L3"
        
        return None, None
    
    async def set(self, key: str, data: dict, ttl: int = None) -> bool:
        """Set data in all cache levels."""
        if ttl is None:
            ttl = config.CACHE_TTL_SECONDS
        
        # Add metadata
        cache_data = {
            **data,
            "timestamp": datetime.now().isoformat(),
            "ttl": ttl
        }
        
        # Level 1: Local cache
        self.local_cache[key] = cache_data
        
        # Level 2: Redis cache
        try:
            redis_data = json.dumps(cache_data)
            success = await self.redis_manager.setex(key, ttl, redis_data)
            if success:
                logger.debug(f"Cache set: Redis (L2) - Key: {key[:16]}...")
            return success
        except Exception as e:
            logger.error(f"Failed to set Redis cache: {e}")
            metrics.record_error("redis")
            return False

multi_cache = MultiLevelCache()

# --------- ROUTER SETUP ---------
router = APIRouter(prefix="/caching", tags=["caching"])

# --------- STARTUP/SHUTDOWN HANDLERS ---------
@asynccontextmanager
async def lifespan(app):
    # Startup
    await redis_manager.initialize()
    yield
    # Shutdown
    await redis_manager.cleanup()

# --------- ENDPOINTS ---------

@router.post("/smart-cache", response_model=PromptResponse)
async def smart_multi_level_cache(
    req: PromptRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Smart multi-level cache with L1 (local) -> L2 (Redis) -> L3 (Database) -> LLM fallback.
    Production-ready with circuit breakers, metrics, and error handling.
    """
    start_time = time.time()
    
    try:
        # Generate cache key
        key = generate_prompt_key(
            req.prompt, 
            req.model_name, 
            req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
        
        # Skip cache if force_refresh is True
        if not req.force_refresh:
            # Try multi-level cache (including L3 database)
            cached_data, cache_level = await multi_cache.get(key, db)
            if cached_data:
                response_time = (time.time() - start_time) * 1000
                return PromptResponse(
                    cached=True,
                    cache_type="smart_multi_level",
                    cache_hit_level=cache_level,
                    response=cached_data["response"],
                    input_tokens=cached_data["input_tokens"],
                    output_tokens=cached_data["output_tokens"],
                    total_tokens=cached_data["input_tokens"] + cached_data["output_tokens"],
                    model=cached_data.get("model", req.model_name),
                    response_time_ms=response_time,
                    finish_reason=cached_data.get("finish_reason", "stop"),
                    usage=cached_data.get("usage", {})
                )
        
        llm_result = await simulate_llm_call(
            req.prompt, 
            req.model_name, 
            req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
        
        background_tasks.add_task(multi_cache.set, key, llm_result, req.ttl_seconds)
        background_tasks.add_task(save_to_database, db, key, req, llm_result)
        
        response_time = (time.time() - start_time) * 1000
        
        return PromptResponse(
            cached=False,
            cache_type="smart_multi_level",
            response=llm_result["response"],
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            total_tokens=llm_result["usage"]["total_tokens"],
            model=llm_result["model"],
            response_time_ms=response_time,
            finish_reason=llm_result["finish_reason"],
            usage=llm_result["usage"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smart cache error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache system error: {str(e)}")

@router.get("/stats", response_model=CacheStats)
async def get_cache_stats():
    """Get comprehensive cache statistics and health metrics."""
    return CacheStats(
        **metrics.get_stats(),
        redis_connected=redis_manager.is_connected,
        local_cache_size=len(local_cache),
        local_cache_maxsize=local_cache.maxsize
    )

@router.post("/clear-cache")
async def clear_cache(cache_level: str = "all", db: Session = Depends(get_db)):
    """Clear cache at specified level(s). Use with caution in production."""
    try:
        if cache_level in ["all", "local", "l1"]:
            local_cache.clear()
            logger.info("Local cache cleared")
        
        if cache_level in ["all", "redis", "l2"] and redis_manager.is_connected:
            logger.warning("Redis cache clear not implemented for safety")
        
        if cache_level in ["all", "database", "l3"]:
            cutoff_time = datetime.now() - timedelta(hours=1)  
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (
                    db.query(PromptCache).filter(PromptCache.timestamp < cutoff_time).delete(),
                    db.commit()
                )
            )
            logger.info("Database cache entries older than 1 hour cleared")
        
        return {"message": f"Cache level '{cache_level}' cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    redis_status = "connected" if redis_manager.is_connected else "disconnected"
    circuit_breaker_status = redis_circuit_breaker.state
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "circuit_breaker": circuit_breaker_status,
        "local_cache_size": len(local_cache),
        "uptime": time.time()
    }

# --------- CACHE WARMING UTILITY ---------
@router.post("/warm-cache")
async def warm_cache(
    prompts: List[str], 
    model_name: str,  
    max_tokens: int = 100,
    background_tasks: BackgroundTasks = None
):
    """
    Warm up the cache with common prompts.
    Useful for production deployment.
    This API preloads the cache with generated responses for common prompts on a given model to speed up future requests. It takes as input the model name, a token limit, and a list of prompts to cache.
    """
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts provided")
    
    if len(prompts) > 100:
        raise HTTPException(status_code=400, detail="Too many prompts (max 100)")
    
    async def warm_single_prompt(prompt: str):
        try:
            req = PromptRequest(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens
            )
            await smart_multi_level_cache(req, BackgroundTasks(), next(get_db()))
        except Exception as e:
            logger.error(f"Cache warming failed for prompt: {e}")
    
    for prompt in prompts:
        if background_tasks:
            background_tasks.add_task(warm_single_prompt, prompt)
        else:
            await warm_single_prompt(prompt)
    
    return {
        "message": f"Cache warming initiated for {len(prompts)} prompts",
        "model": model_name
    }

# --------- CACHE ANALYTICS ENDPOINTS ---------
@router.get("/analytics/top-prompts")
async def get_top_prompts(limit: int = 10, db: Session = Depends(get_db)):
    """Get most frequently cached prompts for analytics."""
    try:
       
        recent_prompts = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: db.query(PromptCache).order_by(PromptCache.timestamp.desc()).limit(limit).all()
        )
        
        return {
            "top_prompts": [
                {
                    "prompt_key": p.prompt_key,
                    "raw_prompt": p.raw_prompt[:100] + "..." if len(p.raw_prompt) > 100 else p.raw_prompt,
                    "model_name": p.model_name,
                    "timestamp": p.timestamp
                }
                for p in recent_prompts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@router.delete("/cleanup/expired")
async def cleanup_expired_cache(db: Session = Depends(get_db)):
    """Cleanup expired cache entries from database."""
    try:
        cutoff_time = datetime.now() - timedelta(seconds=config.LONG_TERM_CACHE_TTL)
        
        deleted_count = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (
                db.query(PromptCache).filter(PromptCache.timestamp < cutoff_time).count(),
                db.query(PromptCache).filter(PromptCache.timestamp < cutoff_time).delete(),
                db.commit()
            )[0]
        )
        
        return {"message": f"Cleaned up {deleted_count} expired cache entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")