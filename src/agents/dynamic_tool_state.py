from langchain.tools import tool, BaseTool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from typing import Any, Callable
import os

def _tool_spec_name(spec: BaseTool | dict[str, Any]) -> str:
    if isinstance(spec, BaseTool):
        return spec.name
    name = spec.get("name")
    if isinstance(name, str):
        return name
    func = spec.get("function")
    if isinstance(func, dict):
        inner = func.get("name")
        if isinstance(inner, str):
            return inner
    return ""

@tool
def public_weather_tool(city: str) -> str:
    """A public tool that is available to all agents."""
    return f"The current weather in {city} is sunny, 42℃"

@tool
def private_weather_tool(city: str) -> str:
    """A tool that is only available to logged-in agents."""
    return f"The current weather in {city} is rainy, 15℃"


model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

@wrap_model_call
def filter_tool_by_state(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to filter tool calls based on agent state."""
    
    is_login = request.state.get("is_login", False)
    
    filter_word = "public" if not is_login else "private"
    tools = [t for t in request.tools if _tool_spec_name(t).startswith(filter_word)]
    request = request.override(tools=tools)
        
    return handler(request)

agent = create_agent(
    model=model,
    tools=[public_weather_tool, private_weather_tool],
    middleware=[filter_tool_by_state],
)

result = agent.invoke(
    {"messages": [HumanMessage("What is the current weather in Beijing?")]}
)
print(result["messages"][-1].content)