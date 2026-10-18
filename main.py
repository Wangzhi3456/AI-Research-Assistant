
# 从 llm_client.py 导入大模型调用函数
from llm_client import ask_llm

# 创建聊天历史列表
messages = []

print("===== AI 智能科研助手 =====")
print("输入 exit 可以退出聊天。\n")

# 持续接收用户输入
while True:

    question = input("你：").strip()

    if question.lower() == "exit":
        print("聊天结束！")
        break

    if not question:
        continue

    # 保存用户问题
    messages.append({
        "role": "user",
        "content": question
    })

    try:
        # 调用 llm_client.py 中的函数
        answer = ask_llm(messages)

        # 显示模型回答
        print("\nAI：", answer, "\n")

        # 保存 AI 的回答
        messages.append({
            "role": "assistant",
            "content": answer
        })

        print("【状态】回答已完成，并保存到聊天历史。\n")

    except Exception as e:

        # 调用失败，撤销本轮用户问题
        messages.pop()

        print("\n【错误】模型调用失败。")
        print("错误信息：", e)
        print("本轮问答没有保存到聊天历史。\n")
