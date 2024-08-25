from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

# Text embeddings and language models (in my case I'm using ollama with local models but you can use others too)
EMBEDDING_MODEL_FUNCTION = OllamaEmbeddings(
    model="bge-m3"
)  # Manually specify your model please
LANG_MODEL_FUNCTION = Ollama(model="llama3.1:latest")

# Query variables
PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based only on the above context: {question}
"""
SOURCE_NUM: int = 10  # How many chunks to take into account when answering a query
