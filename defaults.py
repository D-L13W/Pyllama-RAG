import argparse

# Storage variables
DEFAULT_DATA_PATH: str = "data"
DEFAULT_DB_PATH: str = "chroma"
DEFAULT_SPLIT_METHOD: str = "semantic"
DEFAULT_RECURSIVE_CHUNK_SIZE: int = 400
DEFAULT_RECURSIVE_CHUNK_OVERLAP: int = 40
DEFAULT_SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: int = 10

# Embedding and language model variables
PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based only on the above context: {question}
"""
DEFAULT_NUM_SOURCES: int = 8
DEFAULT_EMBEDDING_MODEL_PROVIDER: str = "ollama"
DEFAULT_EMBEDDING_MODEL: str = "bge-m3"
DEFAULT_LANGUAGE_MODEL_PROVIDER: str = "ollama"
DEFAULT_LANGUAGE_MODEL: str = "phi3:14b-medium-4k-instruct-q4_0"


# Printing settings based on CLI args
def print_settings(args: argparse.Namespace):
    args_dict = vars(args)
    formatting_space = len(max(args_dict.keys(), key=len))
    for key in args_dict:
        print(f"{key:>{formatting_space}} -> {args_dict[key]}")
