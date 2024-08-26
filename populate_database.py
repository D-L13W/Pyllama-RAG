import argparse
import os
import shutil
import chunk_handling
import storage_handling
import model_vars


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    pdf_documents = storage_handling.pdfload()
    chunks = chunk_handling.semantic_split_documents(pdf_documents)
    chunk_handling.add_to_chroma(chunks)


def clear_database():
    if os.path.exists(storage_handling.CHROMA_PATH):
        shutil.rmtree(storage_handling.CHROMA_PATH)


if __name__ == "__main__":
    main()
