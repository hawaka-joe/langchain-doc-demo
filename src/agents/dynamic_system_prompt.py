from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.tools import BaseTool
from typing import Any
from langchain.tools import tool

@tool
def read_data(data: str) -> str:
    """Read the data"""
    return f"Read results for {data}"


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest[Context]) -> str:
    """Generate system prompt based on user role."""
    ctx = request.runtime.context
    if ctx is None:
        return "You are a helpful assistant."
    user_role = ctx.get("user_role", "user")
    base_prompt = "You are a helpful assistant."
    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    else:
        return base_prompt

agent = create_agent(
    model="gemini-3-flash-preview-free",
    tools=[read_data],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)