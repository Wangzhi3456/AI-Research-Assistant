
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载环境变量
load_dotenv()

# 2. 创建大模型客户端
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("YEELO_API_KEY"),
    timeout=30,
    max_retries=0
)


# 3. 核心函数：流式调用大模型
def stream_llm(messages):
    """
    调用大模型，并逐块返回生成的文字。
    """

    finish_reason = None
    has_content = False

    stream = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
        stream=True,
        max_tokens=2000
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        choice = chunk.choices[0]

        # 记录结束原因
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

        # 获取当前生成的文字
        content = choice.delta.content

        if content:
            has_content = True
            yield content

    # 检查是否正常结束
    if finish_reason == "length":
        raise RuntimeError("模型回答达到长度限制，内容可能不完整。")

    if finish_reason != "stop" or not has_content:
        raise RuntimeError("模型没有正常完成回答。")


# 4. 普通函数：获取完整回答
def ask_llm(messages):
    """
    调用 stream_llm，并拼接成完整字符串。
    """

    return "".join(stream_llm(messages))
