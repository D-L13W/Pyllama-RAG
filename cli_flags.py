import settings

# CLI interface flags. `dest` attribute names are specified for clarity though they are normally inferred from the flags themselves; these indicate the attribute names of instantiated parser objects.

split_method_args: list[str] = ["--sm", "--split-method"]
split_method_kwargs: dict = {
    "dest": "split_method",
    "type": str,
    "default": settings.DEFAULT_SPLIT_METHOD,
    "choices": ["recursive", "semantic", "unstructured"],
    "help": "Specifies splitting method. Use langchain's recursive text splitting ('recursive') or the experimental semantic text splitting ('semantic'), or opt for splitting via the unstructured library ('unstructured'). The langchain methods are implemented only for pdfs.",
}

data_path_args: list[str] = ["--data", "--data-path"]
data_path_kwargs: dict = {
    "dest": "data_path",
    "type": str,
    "default": settings.DEFAULT_DATA_PATH,
    "help": "Specifies data path.",
}

db_path_args: list[str] = ["--db", "--db-path"]
db_path_kwargs: dict = {
    "dest": "db_path",
    "type": str,
    "default": settings.DEFAULT_DB_PATH,
    "help": "Specifies database path.",
}

reset_db_args: list[str] = ["--reset", "--reset-db"]
reset_db_kwargs: dict = {
    "dest": "reset_db",
    "action": "store_true",
    "default": False,  # default is False regardless; specified for clarity
    "help": "Resets the database.",
}

query_text_args: list[str] = ["-q", "--query"]
query_text_kwargs: dict = {
    "dest": "query_text",
    "type": str,
    "help": "Specifies query text.",
}

num_sources_args: list[str] = ["-n", "--num-sources"]
num_sources_kwargs: dict = {
    "dest": "num_sources",
    "type": int,
    "default": settings.DEFAULT_NUM_SOURCES,
    "help": "Specifies how many chunks/sources to take into account when answering a query.",
}

embedding_model_provider_args: list[str] = ["--ebmp", "--embedding-model-provider"]
embedding_model_provider_kwargs: dict = {
    "dest": "embedding_model_provider",
    "type": str,
    "default": settings.DEFAULT_EMBEDDING_MODEL_PROVIDER,
    "choices": ["ollama", "anthropic", "openai"],
    "help": f"Specifies embedding model provider to use. Default is '{settings.DEFAULT_EMBEDDING_MODEL_PROVIDER}'",
}

embedding_model_args: list[str] = ["--ebm", "--embedding-model"]
embedding_model_kwargs: dict = {
    "dest": "embedding_model",
    "type": str,
    "default": settings.DEFAULT_EMBEDDING_MODEL,
    "help": f"Specifies embedding model to use in string format. May be a model name, a path to a local model file, etc. depending on the model provider. Default is '{settings.DEFAULT_EMBEDDING_MODEL}'.",
}

language_model_provider_args: list[str] = ["--lmp", "--language-model-provider"]
language_model_provider_kwargs: dict = {
    "dest": "language_model_provider",
    "type": str,
    "default": settings.DEFAULT_LANGUAGE_MODEL_PROVIDER,
    "choices": ["ollama", "anthropic", "openai"],
    "help": f"Specifies language model provider to use. Default is '{settings.DEFAULT_LANGUAGE_MODEL_PROVIDER}'",
}

language_model_args: list[str] = ["--lm", "--language-model"]
language_model_kwargs: dict = {
    "dest": "language_model",
    "type": str,
    "default": settings.DEFAULT_LANGUAGE_MODEL,
    "help": f"Specifies language model to use in string format. May be a model name, a path to a local model file, etc. depending on the model provider. Default is '{settings.DEFAULT_LANGUAGE_MODEL}'.",
}
