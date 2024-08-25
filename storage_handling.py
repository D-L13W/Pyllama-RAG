from langchain_community.document_loaders import PyPDFDirectoryLoader

CHROMA_PATH: str = "chroma"
PDF_DATA_PATH: str = "/Users/stuff/Downloads/Zotero/Zotero_Attachment_Library"


def pdfload():
    return PyPDFDirectoryLoader(PDF_DATA_PATH).load()
