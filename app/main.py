import logging
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.rate_limit import RateLimitMiddleware
from app.api.v1.router import api_router
from app.mcp_server import build_mcp_http_app, mcp as mcp_server

mcp_http_app = build_mcp_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        # The mounted MCP app's session manager needs its own background
        # task running for the lifetime of the process - entering it here
        # (rather than relying on Starlette to start it, which only
        # happens for a sub-app's OWN lifespan, never automatically for a
        # mounted one) is the documented way to combine it with ours.
        await stack.enter_async_context(mcp_server.session_manager.run())

        # Startup
        logging.info(f"Starting {settings.APP_NAME} v1.0")
        yield
        # Shutdown
        logging.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    description="RAG Backend API for document Q&A and benchmarking",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (100 requests per minute for general endpoints)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Exception handlers for consistent ApiResponse
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": f"HTTP_{exc.status_code}", "message": exc.detail}}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": {"code": "VALIDATION_ERROR", "message": "Invalid request data", "details": exc.errors()}}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
    )

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)

# MCP server - same tools as the agent (retrieve, check_deprecation_status,
# answer, finish) exposed over HTTP for any MCP client, local or remote.
app.mount(f"{settings.API_PREFIX}/mcp", mcp_http_app)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )