import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import polars as pl
import model_functions

PROMPT_TEMPLATE: str = """
Answer the question based only on the following context:

{context}

---

Answer the following question based only on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--query",  # keys are inferred from the CLI flags
        dest="query_text",
        type=str,
        help="Specifies query text.",
    )
    parser.add_argument(
        "--db",
        "--db-path",
        dest="db_path",
        type=str,
        default="db",
        help="Specifies database path to read.",
    )
    parser.add_argument(
        "-n",
        "--num-sources",
        dest="num_sources",
        type=int,
        default=6,
        help="Specifies how many chunks/sources to take into account when answering a query.",
    )
    parser.add_argument(
        "--ebm",
        "--embedding-model",
        dest="embedding_model",
        type=str,
        default="bge-m3",
        help="Specifies embedding model to use (via Ollama).",
    )
    parser.add_argument(
        "--lm",
        "--language-model",
        dest="language_model",
        type=str,
        default="phi3:14b-medium-4k-instruct-q4_0",
        help="Specifies language model to use (via Ollama).",
    )
    args = parser.parse_args()

    # Prints a confirmation of CLI arguments
    args_dict = vars(args)
    for key in args_dict:
        print(f"{key} -> {args_dict[key]}")

    # Get model functions based on provided strings
    embedding_model_function = model_functions.get_embed_model_func(
        embedding_model=args.embedding_model
    )
    language_model_function = model_functions.get_lang_model_func(
        language_model=args.language_model
    )

    # Query the database using CLI arguments
    query_db(
        query_text=args.query_text,
        db_path=args.db_path,
        num_sources=args.num_sources,
        embedding_model_function=embedding_model_function,
        language_model_function=language_model_function,
    )


def query_db(
    query_text: str,
    db_path: str,
    num_sources: int,
    embedding_model_function,
    language_model_function,
):
    # Prepare the DB.
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model_function,
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=num_sources)

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
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = language_model_function.invoke(prompt)

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
