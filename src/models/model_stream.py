import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

result = model.stream("how to be rich?")
for chunk in result:
    print(chunk.content, end="", flush=True)
