import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import storage_handling
import model_vars


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(
        persist_directory=storage_handling.CHROMA_PATH,
        embedding_function=model_vars.EMBEDDING_MODEL_FUNCTION,
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=model_vars.NUM_SOURCES)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(model_vars.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = model_vars.LANGUAGE_MODEL_FUNCTION.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = (
        f"Response:\n{response_text}\nSources:\n{'\n'.join(sources)}"
    )
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
