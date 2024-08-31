from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_chroma import Chroma
from unstructured.partition.auto import partition


def recursive_split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def semantic_split_documents(
    documents: list[Document],
    embedding_model_function,
    breakpoint_threshold_amount: int,
):
    text_splitter = SemanticChunker(
        embeddings=embedding_model_function,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )  # uses default 'percentile' threshold type
    return text_splitter.split_documents(documents)


def sync_to_db(chunks: list[Document], db_path: str, embedding_model_function):
    # Load the existing database.
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model_function,
    )

    # Calculate Page IDs.
    chunks_in_data = calculate_chunk_ids(chunks)

    # Get info on chunks in data path
    ids_in_data = set([chunk.metadata["id"] for chunk in chunks_in_data])
    print(f"Number of chunks in data path: {len(ids_in_data)}")

    # Get info on chunks in DB
    chunks_in_db_dict = db.get(include=[])  # IDs are always included by default
    ids_in_db = set(
        chunks_in_db_dict["ids"]
    )  # key is "ids" not "id" because this is chroma's api
    print(f"Number of chunks in DB: {len(ids_in_db)}")

    # Check chunks that don't exist in the DB but do in data path (need to add to DB)
    extra_chunks_in_data = []
    for chunk in chunks_in_data:
        if chunk.metadata["id"] not in ids_in_db:
            # need to have a unique hashable id for each document/chunk
            extra_chunks_in_data.append(chunk)

    # Check chunks that don't exist in data path but do in DB (need to remove from DB)
    extra_chunk_ids_in_db = []
    for chunk_id in chunks_in_db_dict["ids"]:
        if chunk_id not in ids_in_data:
            extra_chunk_ids_in_db.append(chunk_id)

    # Add extra chunks from data path to DB
    if len(extra_chunks_in_data):
        print(f"👉 Adding new chunks: {len(extra_chunks_in_data)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in extra_chunks_in_data]
        db.add_documents(
            extra_chunks_in_data, ids=new_chunk_ids
        )  # here is where you assign your own ids to the documents
        # As you can see Chroma registers the key as "ids", independent of the key you used in chunk.metadata ("id")
    else:
        print("✅ No chunks to add to DB")

    # Remove extra chunks in DB
    if len(extra_chunk_ids_in_db):
        print(f"👉 Removing extra chunks: {len(extra_chunk_ids_in_db)}")
        db.delete(ids=extra_chunk_ids_in_db)
    else:
        print("✅ No chunks to remove from DB")


def calculate_chunk_ids(chunks):

    # This will create unique IDs like "<Page Source>:<Page Index>:<Chunk Index>". It creates this id and adds it together with the chunk index to the metadata of each chunk.

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add ID and chunk index to the page meta-data.
        chunk.metadata["id"] = chunk_id
        chunk.metadata["chunk"] = (
            current_chunk_index  # chunk index only for a given page of a given source
        )

    return chunks
