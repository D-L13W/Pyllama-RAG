from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH: str = "chroma"
PDF_DATA_PATH: str = "pdf_data"
EMBEDDING_CHUNK_SIZE: int = 20000  # in terms of character count
EMBEDDING_CHUNK_OVERLAP: int = 1000
EMBEDDING_FUNCTION = OllamaEmbeddings(
    model="bge-m3"
)  # Manually specify your model please
LANG_MODEL: str = "llama3:latest"
PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based on the above context: {question}
"""
SOURCE_NUM: int = 5
