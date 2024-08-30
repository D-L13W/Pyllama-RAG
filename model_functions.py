from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama


# Can use other providers other than Ollama too; implementation here is specific to Ollama
def get_embed_model_func(embedding_model: str):
    embedding_model_function = OllamaEmbeddings(model=embedding_model)
    return embedding_model_function


def get_lang_model_func(language_model: str):
    language_model_function = Ollama(model=language_model)
    return language_model_function
