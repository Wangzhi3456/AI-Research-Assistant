import streamlit as st
from pathlib import Path

from llm_client import stream_llm
from rag import rag_answer, build_database, get_paper_status

# ===== 1. 网页设置 =====

st.set_page_config(
    page_title="AI 智能科研助手",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI 智能科研助手")
st.caption("基于 RAG 与大语言模型的智能科研问答系统")
st.sidebar.subheader("📚 论文知识库")

uploaded_file = st.sidebar.file_uploader(
    "上传科研论文（PDF）",
    type=["pdf"]
)


# ===== 论文选择 =====

papers_dir = Path("data/papers")
paper_files = sorted(
    p.name for p in papers_dir.glob("*.pdf")
)

if paper_files:
    selected_paper = st.sidebar.selectbox(
        "选择用于问答的论文",
        options=paper_files
    )
else:
    selected_paper = None
    st.sidebar.info("请先上传并建立论文知识库。")



# ===== 保存上传的论文并建立知识库 =====

if uploaded_file is not None:

    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = papers_dir / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.sidebar.success(f"已上传：{uploaded_file.name}")

    if st.sidebar.button("开始建立知识库"):
        with st.spinner("正在解析论文、生成向量并写入数据库..."):
            try:
                build_database(str(pdf_path))
                st.sidebar.success("论文已成功入库！")
            except Exception as e:
                st.sidebar.error(f"入库失败：{e}")


# ===== 论文知识库状态 =====

st.sidebar.divider()
st.sidebar.subheader("📊 入库状态")

# 获取本地保存的所有 PDF 文件
paper_files = sorted(
    p.name for p in papers_dir.glob("*.pdf")
)

if not paper_files:
    st.sidebar.info("目前没有本地论文。")

else:
    for filename in paper_files:

        status = get_paper_status(filename)

        with st.sidebar.expander(filename):

            if status["indexed"]:
                st.success("已入库")
            else:
                st.warning("待入库")

            st.write(
                f"Chunk 数量：{status['chunk_count']}"
            )


# ===== 2. 选择问答模式 =====

mode = st.sidebar.radio(
    "选择问答模式",
    ["普通 LLM 问答", "RAG 论文问答"]
)


# ===== 3. 初始化聊天历史 =====

if "messages" not in st.session_state:
    st.session_state.messages = []

# 保存每轮 RAG 问答对应的检索证据
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = {}

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    st.session_state.messages = []
    st.session_state.rag_sources = {}
    st.session_state.last_mode = mode
    
# ===== 4. 显示历史聊天记录 =====

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # 如果这一轮回答有 RAG 检索证据，就重新显示
        sources = st.session_state.rag_sources.get(i, [])

        if message["role"] == "assistant" and sources:
            with st.expander("查看检索证据"):
                for j, source in enumerate(sources, start=1):
                    st.markdown(f"**证据 {j}**")
                    st.write(f"论文：{source['filename']}")
                    st.write(f"页码：第 {source['page']} 页")

                    if source["distance"] is not None:
                        st.write(
                            f"检索距离：{source['distance']:.4f}"
                        )

                    st.text_area(
                        f"原文片段 {j}",
                        value=source["content"],
                        height=180,
                        disabled=True,
                        key=f"history_source_{i}_{j}"
                    )



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

                # RAG：只检索用户选择的论文
                if selected_paper is None:
                    st.warning("请先上传并建立论文知识库。")

                else:
                    with st.spinner("正在检索指定论文并生成回答..."):
                        result = rag_answer(
                            question,
                            paper_filename=selected_paper,
                            return_sources=True
                        )

                    answer = result["answer"]
                    # 当前即将保存的 assistant 消息对应的索引
                    assistant_index = len(st.session_state.messages)
                    # 将本轮检索证据保存到聊天历史对应的位置
                    st.session_state.rag_sources[assistant_index] = result["sources"]
                    st.write(answer)

                    # 展示实际检索到的论文证据
                    with st.expander("查看检索证据"):
                        if result["sources"]:
                            for i, source in enumerate(result["sources"], start=1):
                                st.markdown(f"**证据 {i}**")
                                st.write(f"论文：{source['filename']}")
                                st.write(f"页码：第 {source['page']} 页")

                                if source["distance"] is not None:
                                    st.write(f"检索距离：{source['distance']:.4f}")

                                st.text_area(
                                    f"原文片段 {i}",
                                    value=source["content"],
                                    height=180,
                                    disabled=True,
                                    key=f"source_{i}"
                                )
                        else:
                            st.info("没有检索到可展示的论文证据。")


        except Exception as e:

            st.session_state.messages.pop()

            st.error(f"模型调用失败：{e}")

        else:

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })