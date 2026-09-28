"""MCP Server: package-registry

Exposes package registry tools for searching and retrieving package information.
This is a simulated package registry for demonstration purposes.
In production, this would connect to a real package registry API (npm, PyPI, etc.).
"""
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("package-registry")


# Simulated package database for demonstration
SAMPLE_PACKAGES = {
    "requests": {
        "name": "requests",
        "version": "2.32.3",
        "description": "Python HTTP for Humans - simple, elegant HTTP library",
        "author": "Kenneth Reitz",
        "license": "Apache-2.0",
        "homepage": "https://requests.readthedocs.io",
        "repository": "https://github.com/psf/requests",
        "keywords": ["http", "api", "client"],
        "dependencies": ["urllib3", "charset-normalizer", "idna", "certifi"],
        "latest_release_date": "2024-11-01",
    },
    "fastapi": {
        "name": "fastapi",
        "version": "0.115.0",
        "description": "FastAPI framework, high performance, easy to learn, fast to code",
        "author": "Sebastián Ramírez",
        "license": "MIT",
        "homepage": "https://fastapi.tiangolo.com",
        "repository": "https://github.com/tiangolo/fastapi",
        "keywords": ["web", "api", "async", "openapi"],
        "dependencies": ["starlette", "pydantic", "uvicorn"],
        "latest_release_date": "2024-10-15",
    },
    "pydantic": {
        "name": "pydantic",
        "version": "2.9.2",
        "description": "Data validation using Python type hints",
        "author": "Samuel Colvin",
        "license": "MIT",
        "homepage": "https://docs.pydantic.dev",
        "repository": "https://github.com/pydantic/pydantic",
        "keywords": ["validation", "serialization", "type-hints"],
        "dependencies": ["pydantic-core", "typing-extensions"],
        "latest_release_date": "2024-10-20",
    },
    "httpx": {
        "name": "httpx",
        "version": "0.27.2",
        "description": "A fully featured HTTP client for Python 3",
        "author": "Tom Christie",
        "license": "BSD-3-Clause",
        "homepage": "https://www.python-httpx.org",
        "repository": "https://github.com/encode/httpx",
        "keywords": ["http", "async", "client", "api"],
        "dependencies": ["h11", "httpcore", "certifi", "idna"],
        "latest_release_date": "2024-10-10",
    },
    "uvicorn": {
        "name": "uvicorn",
        "version": "0.30.6",
        "description": "ASGI server implementation, using uvloop and httptools",
        "author": "Tom Christie",
        "license": "BSD-3-Clause",
        "homepage": "https://www.uvicorn.org",
        "repository": "https://github.com/encode/uvicorn",
        "keywords": ["asgi", "server", "async", "http"],
        "dependencies": ["httptools", "uvloop", "websockets"],
        "latest_release_date": "2024-10-05",
    },
}


@mcp.tool()
async def search_packages(
    query: str,
    limit: int = 10,
) -> dict:
    """Search for packages by name, keyword, or description.

    query: Search term (package name, keyword, or partial description)
    limit: Maximum number of results to return (default 10)

    Returns a list of matching packages with basic info.
    Use get_package_info for full details.
    """
    query_lower = query.lower()
    matches = []

    for pkg in SAMPLE_PACKAGES.values():
        if (
            query_lower in pkg["name"].lower()
            or query_lower in pkg["description"].lower()
            or any(query_lower in kw.lower() for kw in pkg["keywords"])
        ):
            matches.append({
                "name": pkg["name"],
                "version": pkg["version"],
                "description": pkg["description"][:120] + "...",
                "author": pkg["author"],
                "license": pkg["license"],
            })

    return {
        "query": query,
        "total_matches": len(matches),
        "packages": matches[:limit],
    }


@mcp.tool()
async def get_package_info(
    name: str,
    version: Optional[str] = None,
) -> dict:
    """Get detailed information about a specific package.

    name: Exact package name (case-sensitive)
    version: Optional specific version. If omitted, returns latest version info.

    Returns full package metadata including dependencies, repository, etc.
    """
    if name not in SAMPLE_PACKAGES:
        available = ", ".join(sorted(SAMPLE_PACKAGES.keys()))
        return {
            "error": f"Package '{name}' not found in registry",
            "available_packages": available,
        }

    pkg = SAMPLE_PACKAGES[name].copy()

    if version and version != pkg["version"]:
        return {
            "error": f"Version '{version}' not found for package '{name}'",
            "available_version": pkg["version"],
        }

    return pkg


@mcp.tool()
async def check_package_dependencies(
    name: str,
    include_transitive: bool = False,
) -> dict:
    """Check the dependencies of a package.

    name: Package name
    include_transitive: If true, recursively resolve transitive dependencies

    Returns the dependency tree. For this demo, only direct dependencies are available.
    """
    if name not in SAMPLE_PACKAGES:
        available = ", ".join(sorted(SAMPLE_PACKAGES.keys()))
        return {
            "error": f"Package '{name}' not found in registry",
            "available_packages": available,
            "suggestion": f"Try one of the available packages above, or search for similar packages using search_packages(query='xxx')",
        }

    pkg = SAMPLE_PACKAGES[name]
    deps = pkg.get("dependencies", [])

    if not deps:
        return {"package": name, "dependencies": [], "count": 0}

    dep_details = []
    for dep_name in deps:
        if dep_name in SAMPLE_PACKAGES:
            dep_details.append({
                "name": dep_name,
                "version": SAMPLE_PACKAGES[dep_name]["version"],
                "description": SAMPLE_PACKAGES[dep_name]["description"][:80],
            })
        else:
            dep_details.append({
                "name": dep_name,
                "version": "unknown (external)",
                "description": "External dependency not in registry",
            })

    return {
        "package": name,
        "dependencies": dep_details,
        "count": len(dep_details),
        "transitive": include_transitive,
    }


@mcp.tool()
async def list_all_packages(
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List all packages in the registry with pagination.

    limit: Maximum packages to return (default 50, max 100)
    offset: Number of packages to skip (for pagination)

    Returns paginated list of all packages.
    """
    all_packages = list(SAMPLE_PACKAGES.values())
    total = len(all_packages)

    if limit > 100:
        limit = 100

    paginated = all_packages[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "packages": [
            {
                "name": p["name"],
                "version": p["version"],
                "description": p["description"][:100],
                "license": p["license"],
            }
            for p in paginated
        ],
    }


@mcp.tool()
def finish(message: str = "") -> dict:
    """Stop without producing a new answer. Doesn't touch the registry - use it
    purely to end a multi-step research sequence cleanly and leave an optional
    closing note."""
    return {"message": message}


def build_mcp_http_app():
    """Build the ASGI app used to mount this server into the main FastAPI
    app at API_PREFIX + "/mcp/package-registry" - e.g. /api/v1/mcp/package-registry.

    Stateless HTTP + JSON response: every call is a plain HTTP request/response
    with no session affinity or SSE stream to keep alive.
    """
    return mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    mcp.run()