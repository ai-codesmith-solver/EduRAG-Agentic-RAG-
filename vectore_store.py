from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

embedding_model=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


def create_vectore_store(docs):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    if not docs:
        raise ValueError("❌ No documents provided to create vector store.")

    texts = []

    # 🧠 Normalize to list of strings
    for item in docs:
        if isinstance(item, Document):
            texts.append(item.page_content)
        elif isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "page_content" in item:
            texts.append(item["page_content"])
        else:
            print(f"⚠️ Skipping unsupported item: {type(item)}")

    # ✅ Double-check
    if not texts:
        raise ValueError("❌ No valid text found in provided documents.")
    
    chunks=splitter.create_documents(texts)

    vectore_store=FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectore_store