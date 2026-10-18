import pymupdf
import chromadb

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from llm_client import ask_llm


# ===== 1. 初始化模型和数据库 =====

PDF_PATH = "data/papers/test_paper.pdf"

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="research_papers"
)


# ===== 2. PDF 解析 =====

def extract_pdf_text(pdf_path):

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

    pages = extract_pdf_text(pdf_path)

    documents = [
        Document(
            page_content=page["text"],
            metadata={
                "source": pdf_path,
                "page": page["page"]
            }
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

    chunks = split_pdf(pdf_path)

    texts = [chunk.page_content for chunk in chunks]

    vectors = model.encode(
        texts,
        show_progress_bar=True
    )

    collection.upsert(
        ids=[f"test_paper_chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[chunk.metadata for chunk in chunks]
    )

    print(f"论文入库完成！当前共有 {collection.count()} 个 Chunks。")


# ===== 5. 语义检索 =====

def retrieve(question, top_k=3):

    question_vector = model.encode(question)

    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=top_k
    )

    return results


# ===== 6. RAG 问答 =====

def rag_answer(question):

    results = retrieve(question)

    context = ""

    for i, document in enumerate(results["documents"][0]):

        page = results["metadatas"][0][i]["page"]

        context += (
            f"\n[参考资料{i + 1}，第{page}页]\n"
            f"{document}\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名严谨的无线通信科研助手。"
                "请严格根据提供的论文片段回答问题。"
                "注意区分不同技术方案及其对应的实验结论。"
                "不要将一种方法的效果错误归因于另一种方法。"
                "回答时注明参考资料编号和页码。"
                "如果证据不足，请明确说明，不要编造。"
                "请用简洁的中文回答，控制在300字以内。"
            )
        },
        {
            "role": "user",
            "content": f"论文片段：\n{context}\n\n问题：{question}"
        }
    ]

    return ask_llm(messages)


# ===== 7. 主程序 =====

if __name__ == "__main__":

    # 数据库为空时才进行论文入库
    if collection.count() == 0:
        print("数据库为空，开始处理论文...")
        build_database(PDF_PATH)

    question = "高速铁路无线通信中，如何降低切换失败率？"

    print("\n========== RAG 科研问答 ==========")

    answer = rag_answer(question)

    print(answer)
