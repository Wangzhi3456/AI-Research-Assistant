import hashlib
import re
from pathlib import Path

import chromadb
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from llm_client import ask_llm


# ===== 1. 初始化模型和数据库 =====

# 仅供直接运行 rag.py 时使用的默认测试 PDF。
# Streamlit 上传论文时，会将实际路径传给 build_database()。
PDF_PATH = "data/papers/test_paper.pdf"

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="research_papers")


# ===== 2. PDF 解析 =====

def extract_pdf_text(pdf_path):
    """逐页提取 PDF 文本，页码从 1 开始。"""
    pages = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text("text")
            })
    return pages


# ===== 3. 文本切分 =====

def split_pdf(pdf_path):
    """按页提取文本并切分；每个 Chunk 保留原始 PDF 页码。"""
    pages = extract_pdf_text(pdf_path)
    documents = [
        Document(
            page_content=page["text"],
            metadata={"source": str(pdf_path), "page": page["page"]}
        )
        for page in pages
        if page["text"].strip()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# ===== 4. 论文入库 =====

def build_database(pdf_path):
    """解析 PDF、生成 Embedding，并写入 Chroma。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    chunks = split_pdf(str(pdf_path))
    if not chunks:
        raise ValueError("该 PDF 未提取到有效文本，无法入库。")

    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    texts = [chunk.page_content for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True)

    ids = [f"{file_hash}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            **chunk.metadata,
            "paper_id": file_hash,
            "filename": pdf_path.name
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=metadatas
    )

    print(
        f"论文入库完成！文件：{pdf_path.name}；"
        f"本次处理 {len(chunks)} 个 Chunks；"
        f"数据库当前共有 {collection.count()} 个 Chunks。"
    )


# ===== 4.1 查询论文入库状态 =====

def get_paper_status(filename):
    results = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )
    chunk_count = len(results["ids"])
    return {
        "filename": filename,
        "indexed": chunk_count > 0,
        "chunk_count": chunk_count
    }


# ===== 5. 识别用户指定的页码范围 =====

def extract_page_range(question):
    """识别“第20-30页”“第20至30页”“20-30页”等格式。"""
    patterns = [
        r"第\s*(\d+)\s*[-–—~～至到]\s*(\d+)\s*页",
        r"(\d+)\s*[-–—~～至到]\s*(\d+)\s*页"
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            return min(start, end), max(start, end)
    return None


# ===== 6. 检索：普通语义检索 / 完整页码范围检索 =====

def retrieve(question, top_k=3, paper_filename=None):
    """
    普通问题：返回语义相似度最高的 top_k 个 Chunk。
    指定页码范围：返回范围内全部已入库 Chunk，并按页码排序。
    """
    page_range = extract_page_range(question)

    if page_range is not None:
        page_start, page_end = page_range
        filters = [
            {"page": {"$gte": page_start}},
            {"page": {"$lte": page_end}}
        ]
        if paper_filename is not None:
            filters.append({"filename": paper_filename})

        results = collection.get(
            where={"$and": filters},
            include=["documents", "metadatas"]
        )

        records = []
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            try:
                chunk_index = int(doc_id.rsplit("_chunk_", 1)[1])
            except (IndexError, ValueError):
                chunk_index = i

            records.append({
                "id": doc_id,
                "document": results["documents"][i],
                "metadata": metadata,
                "chunk_index": chunk_index
            })

        records.sort(
            key=lambda item: (
                item["metadata"].get("page", 0),
                item["chunk_index"]
            )
        )

        # 保持与 Chroma query() 相同的返回结构，兼容 rag_answer()。
        return {
            "ids": [[item["id"] for item in records]],
            "documents": [[item["document"] for item in records]],
            "metadatas": [[item["metadata"] for item in records]],
            # get() 不提供距离，因此以 None 占位。
            "distances": [[None for item in records]]
        }

    # 未指定页码范围：沿用原来的 Top-K 语义检索。
    question_vector = model.encode(question)
    query_params = {
        "query_embeddings": [question_vector.tolist()],
        "n_results": top_k
    }
    if paper_filename is not None:
        query_params["where"] = {"filename": paper_filename}

    return collection.query(**query_params)


# ===== 7. RAG 问答：返回回答和检索证据 =====

def rag_answer(question, paper_filename=None, return_sources=False):
    page_range = extract_page_range(question)
    results = retrieve(question, paper_filename=paper_filename)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        if page_range is not None:
            message = (
                f"没有在第 {page_range[0]}–{page_range[1]} 页"
                "找到已入库的文本块，请检查所选论文、页码范围和入库状态。"
            )
        else:
            message = "没有检索到相关论文片段，请检查论文是否已入库。"

        if return_sources:
            return {"answer": message, "sources": []}
        return message

    sources = []
    context_parts = []

    for i, document in enumerate(documents):
        metadata = metadatas[i]
        filename = metadata.get(
            "filename",
            metadata.get("source", "未知文件")
        )
        page = metadata.get("page", "未知")

        context_parts.append(
            f"\n[论文：{filename}；第{page}页]\n{document}\n"
        )
        sources.append({
            "filename": filename,
            "page": page,
            "content": document,
            "distance": distances[i] if i < len(distances) else None
        })

    # 指定页码范围时，分批总结，避免一次性把所有原文塞进模型。
    if page_range is not None:
        batch_size = 5
        batch_summaries = []

        for batch_start in range(0, len(context_parts), batch_size):
            batch_context = "\n".join(
                context_parts[batch_start:batch_start + batch_size]
            )
            batch_number = batch_start // batch_size + 1
            batch_total = (
                len(context_parts) + batch_size - 1
            ) // batch_size

            batch_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一名严谨的无线通信科研助手。"
                        "你正在分批阅读论文的指定页码范围。"
                        "请只根据本批提供的原文，提取关键主题、技术方法、"
                        "工作原理、重要结论及对应 PDF 页码。"
                        "保留重要技术术语，不要编造或把不同方法混为一谈。"
                        "这是中间整理结果，不需要写成完整文章。"
                        "用中文简洁分点总结，控制在 250 字以内。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n"
                        f"这是第 {batch_number}/{batch_total} 批原文：\n"
                        f"{batch_context}"
                    )
                }
            ]

            batch_summaries.append(ask_llm(batch_messages))

        combined_summaries = "\n\n".join(
            f"第{i}批摘要：\n{summary}"
            for i, summary in enumerate(batch_summaries, start=1)
        )

        final_messages = [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的无线通信科研助手。"
                    "下面是对用户指定 PDF 页码范围分批阅读后得到的摘要。"
                    "请综合所有批次，回答用户的问题。"
                    "覆盖主要主题、关键技术、技术原理和重要结论，"
                    "不要只采用某一批的信息，也不要重复堆砌。"
                    "保留摘要中可确认的 PDF 页码引用。"
                    "如果摘要之间存在差异，请谨慎表述，不要自行编造。"
                    "用中文分点回答，内容完整但避免冗长。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"各批次摘要如下：\n{combined_summaries}"
                )
            }
        ]

        answer = ask_llm(final_messages)

    else:
        # 普通问题仍然使用原来的 Top-K 语义检索逻辑。
        context = "\n".join(context_parts)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的无线通信科研助手。"
                    "请严格根据提供的论文片段回答问题。"
                    "注意区分不同技术方案及其对应的实验结论。"
                    "不要将一种方法的效果错误归因于另一种方法。"
                    "回答时注明论文文件名和 PDF 页码。"
                    "如果证据不足，请明确说明，不要编造。"
                    "请用简洁的中文回答，控制在300字以内。"
                )
            },
            {
                "role": "user",
                "content": f"论文片段：\n{context}\n\n问题：{question}"
            }
        ]
        answer = ask_llm(messages)

    if return_sources:
        return {"answer": answer, "sources": sources}
    return answer


# ===== 8. 主程序：仅用于直接运行 rag.py 时测试 =====

if __name__ == "__main__":
    if collection.count() == 0:
        print("数据库为空，开始处理默认测试论文...")
        build_database(PDF_PATH)

    question = "高速铁路无线通信中，如何降低切换失败率？"
    print("\n========== RAG 科研问答 ==========")
    print(rag_answer(question))
