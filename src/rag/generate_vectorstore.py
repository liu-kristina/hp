import os
import logging
from pathlib import Path
from typing import List
import warnings

from tqdm import tqdm
from faiss import IndexFlatL2

from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# TODO: Re-Generate vectorstore functionality

warnings.simplefilter("ignore", DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

# PyCharm
# FILE_PATH = Path("..", "..", "data", "test")
# STORE_PATH = Path("..", "..", "models", "vectorstore")

FILE_PATH = Path("data", "raw")
STORE_PATH = Path("models", "vectorstore")


def get_files(path: Path) -> list:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdf")]


def init_embeddings():
    logger.info('Initializing embeddings')
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def init_vecstore():

    embeddings = init_embeddings()
    index = IndexFlatL2(len(embeddings.embed_query("hello world")))
    logger.info("Initializing vectorstore")
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store

def add_document(vector_store: VectorStore, doc: Document):

    if not isinstance(doc, list):
        doc = [doc]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    logging.info("Adding PDF file to vectorstore...")
    chunks = text_splitter.split_documents(doc)
    vector_store.add_documents(chunks)
    logging.info("PDF file added to vectorstore")
    # return vector_store

# def _chunk_doc(data, splitter):
#     chunks = splitter.split_documents(data)
#     ids = []
#     for idx, chunk in enumerate(chunks):
#         doc_id = f"{'_'.join(file.stem.split())}_chunk{idx}"
#         ids.append(doc_id)


def populate_vecstore(vector_store: VectorStore, files: List | Path | str):

    if not isinstance(files, list):
        files = [files]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    logging.info("Loading and adding PDF files to vectorstore...")
    for file in tqdm(files):
        file = Path(file)

        # Load PDF file
        loader = PyMuPDFLoader(file)
        data = loader.load()

        chunks = text_splitter.split_documents(data)
        ids = []
        for idx, chunk in enumerate(chunks):
            doc_id = f"{'_'.join(file.stem.split())}_chunk{idx}"
            ids.append(doc_id)

        # Check for duplicates and skip
        if len(vector_store.get_by_ids(ids)) > 0:
            continue
        vector_store.add_documents(chunks, ids=ids)
    logging.info("PDF files added to vectorstore")

def save_vectorstore(vector_store: FAISS, store_path: Path | str, fname: str = "faiss_index"):

    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    fpath = store_path / fname
    logging.info("Saving vectorstore to %s...", fpath)
    vector_store.save_local(str(fpath))
    logging.info("Vectorstore saved to %s.", fpath)

def load_vectorstore(store_path: Path | str, fname: str = "faiss_index"):
    logger.info("Loading vectorstore from '%s'", store_path)
    embeddings = init_embeddings()
    vector_store = FAISS.load_local(str(store_path / fname),
                                    embeddings=embeddings,
                                    allow_dangerous_deserialization=True)
    return vector_store

def create_vectorstore(store_path: Path | str, fname: str = "faiss_index"):
    logger.info("Creating new vectorstore...")
    vector_store = init_vecstore()
    files = get_files(FILE_PATH)
    populate_vecstore(vector_store, files)
    logger.debug("Vectorstore populated.")
    save_vectorstore(vector_store, store_path, fname)
    logger.info("New Vectorstore saved to %s.", store_path)
    return vector_store

def get_vectorstore(store_path: Path | str = STORE_PATH, fname: str = "faiss_index"):
    logger.info("Trying to get vectorstore from %s...", store_path)
    store_path = Path(store_path)
    if store_path.exists():
        vector_store = load_vectorstore(store_path, fname)
        logger.info("Vectorstore loaded from %s.", store_path)
    else:
        vector_store = create_vectorstore(store_path, fname)
        logger.info("New Vectorstore created at %s.", store_path)
    return vector_store

def get_retriever(vector_store: VectorStore):
    return vector_store.as_retriever()


if __name__ == "__main__":

    pdf_path = Path("..", "..", "data", "raw")
    vec_path = Path("models", "vectorstore")

    vec_store = get_vectorstore(vec_path)
    retriever = vec_store.as_retriever()
    docs = retriever.invoke("mini workstation")
    for doc in docs:
        print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')