import os
from typing import Any, cast

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict[str, str]


model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

agent = create_agent(
    model=model,
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    cast(
        Any,
        {
            "messages": [HumanMessage("Hello")],
            "user_id": "user_123",
            "preferences": {"theme": "dark"},
        },
    ),
    {"configurable": {"thread_id": "1"}},
)
print(result["messages"][-1].content)