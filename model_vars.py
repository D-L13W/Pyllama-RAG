from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

# Text embeddings and language models (in my case I'm using ollama with local models but you can use others too)
EMBEDDING_MODEL_FUNCTION = OllamaEmbeddings(
    model="bge-m3"
)  # Manually specify your model please
LANGUAGE_MODEL_FUNCTION = Ollama(model="llama3.1:latest")

# Query variables
PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based only on the above context: {question}
"""
## How many chunks/sources to take into account when answering a query
NUM_SOURCES: int = 10
