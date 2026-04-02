from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Any

@tool
def read_data(data: str) -> str:
    """Read the data"""
    return f"Read results for {data}"

@tool
def write_data(data: str) -> str:
    """Write the data"""
    return f"Write results for {data}"

@tool
def delete_data(data: str) -> str:
    """Delete the data"""
    return f"Delete results for {data}"

def _tool_name(t: BaseTool | dict[str, Any]) -> str:
    if isinstance(t, dict):
        n = t.get("name")
        return str(n) if n is not None else ""
    return t.name

@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(
    request: ModelRequest[Context],
    handler: Callable[[ModelRequest[Context]], ModelResponse],
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    # Read from Runtime Context: get user role
    ctx = request.runtime.context
    if ctx is None:
        # If no context provided, default to viewer (most restrictive)
        user_role = "viewer"
    else:
        user_role = ctx.user_role

    if user_role == "admin":
        # Admins get all tools
        pass
    elif user_role == "editor":
        # Editors can't delete
        tools = [t for t in request.tools if _tool_name(t) != "delete_data"]
        request = request.override(tools=tools)
    else:
        # Viewers get read-only tools
        tools = [t for t in request.tools if _tool_name(t).startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4.1",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)