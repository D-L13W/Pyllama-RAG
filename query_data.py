import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import polars as pl
import settings
import cli_flags


def main():
    # CLI setup
    parser = argparse.ArgumentParser()
    parser.add_argument(*cli_flags.db_path_args, **cli_flags.db_path_kwargs)
    parser.add_argument(
        *cli_flags.embedding_model_provider_args,
        **cli_flags.embedding_model_provider_kwargs,
    )
    parser.add_argument(
        *cli_flags.embedding_model_args,
        **cli_flags.embedding_model_kwargs,
    )
    parser.add_argument(
        *cli_flags.language_model_provider_args,
        **cli_flags.language_model_provider_kwargs,
    )
    parser.add_argument(
        *cli_flags.language_model_args,
        **cli_flags.language_model_kwargs,
    )
    parser.add_argument(*cli_flags.num_sources_args, **cli_flags.num_sources_kwargs)
    parser.add_argument(*cli_flags.query_text_args, **cli_flags.query_text_kwargs)
    args = parser.parse_args()

    # Prints a confirmation of CLI arguments
    args_dict = vars(args)
    formatting_space = len(max(args_dict.keys(), key=len))
    for key in args_dict:
        print(f"{key:>{formatting_space}} -> {args_dict[key]}")

    # Get model functions based on CLI arguments (use defaults in model_functions if unspecified); can use more if else statements
    embedding_model_function = settings.get_embed_model_func(
        provider=args.embedding_model_provider, embedding_model=args.embedding_model
    )
    language_model_function = settings.get_lang_model_func(
        provider=args.language_model_provider, language_model=args.language_model
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
    prompt_template = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
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
