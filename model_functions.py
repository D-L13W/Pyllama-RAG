from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama


# Can use other providers other than Ollama too; implementation here is specific to Ollama; string input to specify the models are inputted by CLI arguments in `query_data` and `refresh_db`
def ollama_get_embed_model_func(embedding_model: str = "bge-m3"):
    embedding_model_function = OllamaEmbeddings(model=embedding_model)
    return embedding_model_function


def ollama_get_lang_model_func(
    language_model: str = "phi3:14b-medium-4k-instruct-q4_0",
):
    language_model_function = Ollama(model=language_model)
    return language_model_function
