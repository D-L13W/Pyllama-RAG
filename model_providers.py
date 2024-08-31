from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama


# Can use other providers other than Ollama too
def get_embed_model_func(provider: str, embedding_model: str):
    if provider == "ollama":
        embedding_model_function = OllamaEmbeddings(model=embedding_model)
    return embedding_model_function


def get_lang_model_func(provider: str, language_model: str):
    if provider == "ollama":
        language_model_function = Ollama(model=language_model)
    return language_model_function
