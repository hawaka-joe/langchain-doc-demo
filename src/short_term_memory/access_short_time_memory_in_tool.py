from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from typing import Any, cast
import os

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model=model,
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke(cast(Any, {
    "messages": "look up user information",
    "user_id": "user_123"
}))
print(result["messages"][-1].content)
# > User is John Smith.