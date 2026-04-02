import os
from collections.abc import Sequence
from typing import Any, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState
    tools: Sequence[BaseTool] = ()

    def before_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(state["user_preferences"])
        print(runtime)
        return None

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

agent = create_agent(
    model,
    tools=[],
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
invoke_input: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "I prefer technical explanations"},
        {"role": "user", "content": "what is RMA"}
    ],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
}
result = agent.invoke(cast(Any, invoke_input))
print(result["messages"][-1].content)