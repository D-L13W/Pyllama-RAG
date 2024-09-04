import argparse
import os
import shutil
import split_methods
from langchain_community.embeddings.ollama import OllamaEmbeddings
import defaults
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
    parser.add_argument(
        *cli_flags.recursive_chunk_size_args, **cli_flags.recursive_chunk_size_kwargs
    )
    parser.add_argument(
        *cli_flags.recursive_chunk_overlap_args,
        **cli_flags.recursive_chunk_overlap_kwargs,
    )
    parser.add_argument(
        *cli_flags.semantic_breakpoint_threshold_amount_args,
        **cli_flags.semantic_breakpoint_threshold_amount_kwargs,
    )
    parser.add_argument(*cli_flags.reset_db_args, **cli_flags.reset_db_kwargs)
    args = parser.parse_args()

    # Check that data path exists
    if not os.path.exists(args.data_path):
        raise Exception("Data path does not exist. Use a valid data path.")

    # Logic after getting CLI arguments
    defaults.print_settings(args=args)
    check_reset_db(args=args)
    split_methods.exec_split_method(args=args)


def check_reset_db(args: argparse.Namespace):
    if args.reset_db:
        print(f"\n🧹 Clearing Database -> {args.db_path}\n")
        if os.path.exists(args.db_path):
            shutil.rmtree(args.db_path)
        else:
            print("Database does not exist. Use a valid data path.")


if __name__ == "__main__":
    main()
