from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama


# Main functions; returns embedding and language model instances based on model name. Can use providers other than Ollama too.
EMBEDDING_MODEL_PROVIDER_CHOICES: list[str] = ["ollama", "anthropic", "openai"]


def get_embed_model_func(provider: str, embedding_model: str):
    if provider == "ollama":
        embedding_model_function = OllamaEmbeddings(model=embedding_model)
    return embedding_model_function


LANGUAGE_MODEL_PROVIDER_CHOICES: list[str] = ["ollama", "anthropic", "openai"]


def get_lang_model_func(provider: str, language_model: str):
    if provider == "ollama":
        language_model_function = Ollama(model=language_model)
    return language_model_function
