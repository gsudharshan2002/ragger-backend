"""MCP Client wrapper for connecting to HTTP Streamable MCP servers.

Allows the Agent to dynamically discover and call tools from
configured MCP servers without hardcoding tool implementations.

Supports:
- Multiple MCP servers (aggregate tool registry)
- Dynamic tool discovery via tools/list
- Tool calls via tools/call
- Zero-code addition of new servers (config only)
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from mcp import Client, ListToolsResult
from mcp.types import (
    Tool as MCPTool,
    CallToolResult,
    TextContent,
)


class MCPClientWrapper:
    """Wrapper around MCP Client for HTTP Streamable transport.

    Handles connection, tool discovery, and tool calls to a single
    MCP server. The Agent can query available tools and invoke them
    dynamically without knowing their implementations.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._client: Optional[Client] = None
        self._tools: Dict[str, MCPTool] = {}
        self._connected: bool = False

    async def __aenter__(self) -> "MCPClientWrapper":
        self._client = Client(self.server_url)
        await self._client.__aenter__()
        await self._discover_tools()
        self._connected = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.__aexit__(*args)
        self._client = None
        self._tools = {}
        self._connected = False

    async def _discover_tools(self) -> None:
        if not self._client:
            raise RuntimeError("MCP client not initialized")
        result: ListToolsResult = await self._client.list_tools()
        self._tools = {tool.name: tool for tool in result.tools}

    def get_tools(self) -> Dict[str, MCPTool]:
        return dict(self._tools)

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        if not self._client or not self._connected:
            raise RuntimeError("MCP client not connected")
        result: CallToolResult = await self._client.call_tool(tool_name, arguments)
        return result

    @property
    def tools_available(self) -> bool:
        return len(self._tools) > 0


class MCPManager:
    """Manages multiple MCP server connections and aggregates all available tools.

    The Agent uses this to discover and call tools from configured MCP servers.
    Adding a new MCP server only requires adding it to the configuration -
    NO Agent source code changes are needed.

    Architecture:
      MCP Host / Client
             |
      +----+----+----+
      |           |
      v           v
  ragger-tools  package-registry
  MCP Server    MCP Server

    The Agent dynamically discovers tools from ALL registered servers.
    The system prompt is built from the live tool list at runtime.
    """

    def __init__(self):
        self._servers: List[MCPClientWrapper] = []
        self._initialized: bool = False

    def add_server(self, name: str, url: str):
        """Add an MCP server by name and URL.

        Args:
            name: Human-readable name for logging
            url: HTTP URL of the MCP server (e.g. http://localhost:8000/api/v1/mcp)
        """
        from mcp import Client

        client = MCPClientWrapper(url)
        self._servers.append(client)

    async def initialize(self) -> bool:
        """Initialize all configured MCP servers and aggregate tools.

        Returns True if at least one server connected successfully.
        """
        self._initialized = False
        all_tools: Dict[str, MCPTool] = {}

        for client in self._servers:
            try:
                async with client:
                    for name, tool in client.get_tools().items():
                        all_tools[name] = tool
            except Exception as e:
                from app.core.logging_config import get_logger
                logger = get_logger(__name__)
                logger.warning(
                    f"MCP server failed to initialize ({client.server_url}): {e}"
                )

        self._initialized = len(all_tools) > 0
        return self._initialized

    async def discover_tools(self) -> Dict[str, MCPTool]:
        """Discover tools from all initialized MCP servers.

        Must be called after initialize(). Returns aggregated tool registry.
        """
        if not self._initialized:
            raise RuntimeError("MCP Manager not initialized")
        all_tools: Dict[str, MCPTool] = {}

        for client in self._servers:
            try:
                async with client:
                    for name, tool in client.get_tools().items():
                        all_tools[name] = tool
            except Exception:
                continue

        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Optional[CallToolResult]:
        """Call a tool by name on the first MCP server that provides it.

        Args:
            tool_name: Name of the tool to call (must match what was discovered)
            arguments: Tool-specific arguments

        Returns:
            CallToolResult or None if no server provides this tool
        """
        if not self._initialized:
            raise RuntimeError("MCP Manager not initialized")

        for client in self._servers:
            try:
                async with client:
                    if tool_name in client.get_tools():
                        return await client.call_tool(tool_name, arguments)
            except Exception:
                continue

        return None

    @property
    def tool_names(self) -> List[str]:
        if not self._initialized:
            return []
        names: List[str] = []
        for client in self._servers:
            names.extend(client.get_tool_names())
        return names

    async def is_healthy(self) -> bool:
        """Check if any MCP server is reachable and healthy."""
        if not self._initialized:
            return False
        for client in self._servers:
            try:
                async with client:
                    if client.tools_available:
                        return True
            except Exception:
                continue
        return False


_manager: Optional[MCPManager] = None


async def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None or not _manager._initialized:
        _manager = MCPManager()
        _manager.add_server("ragger-tools", "http://localhost:8000/api/v1/mcp")
        _manager.add_server("package-registry", "http://localhost:8000/api/v1/mcp/package-registry")
        await _manager.initialize()
    return _manager