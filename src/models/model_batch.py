import os
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

"""
特点
流式/增量返回：哪个先完成就先返回哪个
顺序不保证：输出顺序 ≠ 输入顺序（取决于响应速度）
返回类型：迭代器（generator）
"""
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)

"""
特点
阻塞执行：必须等所有请求都完成后才返回
顺序固定：输出顺序 = 输入顺序
返回类型：列表（list）
"""
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)