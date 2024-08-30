import argparse
import os
import shutil
import chunk_handling
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
import model_functions


def main():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",  # keys are inferred from the CLI flags; dest attributes are also specified for clarity (they are normally inferred from the flag name)
        dest="reset",
        action="store_true",
        default=False,
        help="Resets the database.",
    )  # default is false, but specified just for clarity
    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        type=str,
        default="semantic",
        choices=["recursive", "semantic", "unstructured"],
        help="Specifies splitting method. Use langchain's recursive text splitting ('recursive') or the experimental semantic text splitting ('semantic'), or opt for splitting via the unstructured library ('unstructured'). The langchain methods are implemented only for pdfs.",
    )
    parser.add_argument(
        "--data",
        "--data-path",
        dest="data_path",
        type=str,
        default="data",
        help="Specifies data path to read.",
    )
    parser.add_argument(
        "--db",
        "--db-path",
        dest="db_path",
        type=str,
        default="db",
        help="Specifies database path to sync chunks to.",
    )
    parser.add_argument(
        "--ebm",
        "--embedding-model",
        dest="embedding_model",
        type=str,
        default="bge-m3",
        help="Specifies embedding model to use (via Ollama).",
    )
    args = parser.parse_args()

    # Check that data path exists
    if not os.path.exists(args.data_path):
        raise Exception("Data path does not exist. Use a valid data path.")

    # Prints a confirmation of CLI arguments
    args_dict = vars(args)
    for key in args_dict:
        print(f"{key} -> {args_dict[key]}")

    # Instantiate embedding model
    embedding_model_function = model_functions.get_embed_model_func(
        embedding_model=args.embedding_model
    )

    # Reset database if --reset flag specified
    if args.reset:
        print(f"\n🧹 Clearing Database -> {args.db_path}\n")
        clear_database(db_path=args.db_path)

    # Database creation/sync method
    if args.split == "recursive":
        pdf_documents = pdf_load(data_path=args.data_path)
        chunks = chunk_handling.recursive_split_documents(documents=pdf_documents)
        chunk_handling.sync_to_db(
            chunks=chunks,
            db_path=args.db_path,
            embedding_model_function=embedding_model_function,
        )
    elif args.split == "semantic":
        pdf_documents = pdf_load(data_path=args.data_path)
        chunks = chunk_handling.semantic_split_documents(
            documents=pdf_documents, embedding_model_function=embedding_model_function
        )
        chunk_handling.sync_to_db(
            chunks=chunks,
            db_path=args.db_path,
            embedding_model_function=embedding_model_function,
        )
    elif args.split == "unstructured":
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
