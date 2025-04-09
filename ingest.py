import os
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
DATA_PATH = "Material_Science/"
DB_FAISS_PATH = "vectorstore/db_faiss_MS"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]

# ------------------------------
# Split documents into chunks
# ------------------------------
def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# ------------------------------
# Load documents and build vector DB
# ------------------------------
def create_vector_db():
    loaders = [
        DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    ]

    all_documents = []
    for loader in loaders:
        documents = loader.load()
        all_documents += documents

    docs_processed = split_documents(
        chunk_size=512,
        knowledge_base=all_documents,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = FAISS.from_documents(
        docs_processed, embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
