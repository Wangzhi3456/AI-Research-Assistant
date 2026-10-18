import streamlit as st

from llm_client import stream_llm
from rag import rag_answer


# ===== 1. 网页设置 =====

st.set_page_config(
    page_title="AI 智能科研助手",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI 智能科研助手")
st.caption("基于 RAG 与大语言模型的智能科研问答系统")


# ===== 2. 选择问答模式 =====

mode = st.sidebar.radio(
    "选择问答模式",
    ["普通 LLM 问答", "RAG 论文问答"]
)


# ===== 3. 初始化聊天历史 =====

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

# 切换模式时清空旧聊天，避免两种模式的对话混在一起
if st.session_state.last_mode != mode:
    st.session_state.messages = []
    st.session_state.last_mode = mode


# ===== 4. 显示历史聊天记录 =====

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ===== 5. 接收用户问题 =====

question = st.chat_input("请输入你的科研问题...")

if question:

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # ===== 6. 根据模式生成回答 =====

    with st.chat_message("assistant"):

        try:

            if mode == "普通 LLM 问答":

                # 普通 DeepSeek 对话
                answer = st.write_stream(
                    stream_llm(st.session_state.messages)
                )

            else:

                # RAG：先检索论文，再调用 DeepSeek
                with st.spinner("正在检索论文并生成回答..."):
                    answer = rag_answer(question)

                st.write(answer)

        except Exception as e:

            st.session_state.messages.pop()

            st.error(f"模型调用失败：{e}")

        else:

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })