import argparse
import os
import shutil
import split_methods
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
import settings
import cli_flags


def main():
    # CLI setup
    parser = argparse.ArgumentParser()
    parser.add_argument(*cli_flags.data_path_args, **cli_flags.data_path_kwargs)
    parser.add_argument(*cli_flags.db_path_args, **cli_flags.db_path_kwargs)
    parser.add_argument(
        *cli_flags.embedding_model_provider_args,
        **cli_flags.embedding_model_provider_kwargs,
    )
    parser.add_argument(
        *cli_flags.embedding_model_args,
        **cli_flags.embedding_model_kwargs,
    )
    parser.add_argument(*cli_flags.split_method_args, **cli_flags.split_method_kwargs)
    parser.add_argument(*cli_flags.reset_db_args, **cli_flags.reset_db_kwargs)
    args = parser.parse_args()

    # Check that data path exists
    if not os.path.exists(args.data_path):
        raise Exception("Data path does not exist. Use a valid data path.")

    # Prints a confirmation of CLI arguments
    args_dict = vars(args)
    formatting_space = len(max(args_dict.keys(), key=len))
    for key in args_dict:
        print(f"{key:>{formatting_space}} -> {args_dict[key]}")

    # Get model functions based on CLI arguments (use defaults in model_functions if unspecified)
    embedding_model_function = settings.get_embed_model_func(
        provider=args.embedding_model_provider, embedding_model=args.embedding_model
    )

    # Reset database if --reset flag specified
    if args.reset_db:
        print(f"\n🧹 Clearing Database -> {args.db_path}\n")
        clear_database(db_path=args.db_path)

    # Database creation/sync method
    if args.split_method == "recursive":
        pdf_documents = pdf_load(data_path=args.data_path)
        chunks = split_methods.recursive_split_documents(documents=pdf_documents)
        split_methods.sync_to_db(
            chunks=chunks,
            db_path=args.db_path,
            embedding_model_function=embedding_model_function,
        )
    elif args.split_method == "semantic":
        pdf_documents = pdf_load(data_path=args.data_path)
        chunks = split_methods.semantic_split_documents(
            documents=pdf_documents, embedding_model_function=embedding_model_function
        )
        split_methods.sync_to_db(
            chunks=chunks,
            db_path=args.db_path,
            embedding_model_function=embedding_model_function,
        )
    elif args.split_method == "unstructured":
        documents = all_file_load(data_path=args.data_path)


def clear_database(db_path: str):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def pdf_load(data_path: str):
    return PyPDFDirectoryLoader(data_path).load()


def all_file_load(data_path: str):
    filepaths = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(data_path)
        for f in filenames
    ]
    return filepaths


if __name__ == "__main__":
    main()
