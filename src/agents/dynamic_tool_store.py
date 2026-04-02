from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore


def _tool_name(t: BaseTool | dict[str, Any]) -> str:
    if isinstance(t, dict):
        n = t.get("name")
        return str(n) if n is not None else ""
    return t.name

@tool
def search_tool(query: str) -> str:
    """Search the web for information"""
    return f"Search results for {query}"

@tool
def analysis_tool(data: str) -> str:
    """Analyze the data"""
    return f"Analysis results for {data}"

@tool
def export_tool(data: str) -> str:
    """Export the data"""
    return f"Export results for {data}"

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(
    request: ModelRequest[Context],
    handler: Callable[[ModelRequest[Context]], ModelResponse],
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    ctx = request.runtime.context
    if ctx is None:
        return handler(request)
    user_id = ctx.user_id

    store = request.runtime.store
    if store is None:
        return handler(request)

    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        tools = [t for t in request.tools if _tool_name(t) in enabled_features]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gemini-3-flash-preview-free",
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)