import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import polars as pl
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

    sources = pl.DataFrame(
        {
            "content": [doc.page_content for doc, _score in results],
            "source": [
                doc.metadata.get("source", None).split("/")[-1]
                for doc, _score in results
            ],  # filename
            "page": [doc.metadata.get("page", None) for doc, _score in results],
            "chunk": [doc.metadata.get("chunk", None) for doc, _score in results],
        }
    )  # Results list is small enough that this is fine
    context_text = "\n\n---\n\n".join(sources["content"])
    prompt_template = ChatPromptTemplate.from_template(model_vars.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = model_vars.LANGUAGE_MODEL_FUNCTION.invoke(prompt)

    with pl.Config(
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_width_chars=160,
        set_fmt_str_lengths=80,
    ):
        formatted_response = f"Response:\n\n{response_text}\n\nSources:\n\n{sources}"
        print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
