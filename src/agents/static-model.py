import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 这里为什么使用的是 gemini 模型，但是模型 provider 是 openai？
# model_provider="openai" 指的不是“模型是谁”，而是 “你用的是 OpenAI 兼容协议”
# aihubmix 之所以让你用 openai 包，是因为它提供的是 OpenAI 兼容接口
# 你通常不能直接用 Gemini 官方 SDK 去调用 aihubmix，除非它也提供 Gemini 兼容协议（一般不会）
model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

agent = create_agent(model=model)

result = agent.invoke(
    {"messages": [HumanMessage("What is the capital of France?")]}
)
print(result["messages"][-1].content)
