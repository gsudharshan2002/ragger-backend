import time
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP address)."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, client_id: str, now: float):
        """Remove requests older than 1 minute."""
        cutoff = now - 60
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff
        ]
    
    def is_allowed(self, request: Request) -> bool:
        """Check if request is allowed."""
        client_id = self._get_client_id(request)
        now = time.time()
        
        self._clean_old_requests(client_id, now)
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_retry_after(self, request: Request) -> int:
        """Get seconds until next request is allowed."""
        client_id = self._get_client_id(request)
        if not self.requests[client_id]:
            return 0
        
        oldest = min(self.requests[client_id])
        return max(0, int(60 - (time.time() - oldest)))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to apply rate limiting to all requests."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute)
    
    async def dispatch(self, request: Request, call_next):
        if not self.limiter.is_allowed(request):
            retry_after = self.limiter.get_retry_after(request)
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(retry_after)}
            )
        
        response = await call_next(request)
        return response


# Chat-specific rate limiter (stricter)
chat_rate_limiter = RateLimiter(requests_per_minute=30, burst_size=5)


def check_chat_rate_limit(request: Request):
    """Check rate limit for chat endpoints."""
    if not chat_rate_limiter.is_allowed(request):
        retry_after = chat_rate_limiter.get_retry_after(request)
        raise HTTPException(
            status_code=429,
            detail="Too many chat requests. Please wait before sending another message.",
            headers={"Retry-After": str(retry_after)}
        )
