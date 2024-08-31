from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import argparse

# Storage variables
DEFAULT_DATA_PATH: str = "data"
DEFAULT_DB_PATH: str = "db"
DEFAULT_SPLIT_METHOD: str = "semantic"

# Embedding and language model variables
PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based only on the above context: {question}
"""
DEFAULT_NUM_SOURCES: int = 8
DEFAULT_EMBEDDING_MODEL_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_LANGUAGE_MODEL_PROVIDER = "ollama"
DEFAULT_LANGUAGE_MODEL = "phi3:14b-medium-4k-instruct-q4_0"


def print_settings(args: argparse.Namespace):
    args_dict = vars(args)
    formatting_space = len(max(args_dict.keys(), key=len))
    for key in args_dict:
        print(f"{key:>{formatting_space}} -> {args_dict[key]}")


# Can use other providers other than Ollama too
def get_embed_model_func(provider: str, embedding_model: str):
    if provider == "ollama":
        embedding_model_function = OllamaEmbeddings(model=embedding_model)
    return embedding_model_function


def get_lang_model_func(provider: str, language_model: str):
    if provider == "ollama":
        language_model_function = Ollama(model=language_model)
    return language_model_function
