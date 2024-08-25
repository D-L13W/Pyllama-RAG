from langchain_community.document_loaders import PyPDFDirectoryLoader

CHROMA_PATH: str = "chroma"
PDF_DATA_PATH: str = "data"


def pdfload():
    return PyPDFDirectoryLoader(PDF_DATA_PATH).load()
