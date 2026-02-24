import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

def load_documents_from_directory(directory_path):
    """Load all .txt files from the specified directory."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    loader = DirectoryLoader(path=directory_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text files found in directory '{directory_path}'.")

    print(f"Loaded {len(documents)} documents from '{directory_path}'.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller overlapping chunks for better retrieval."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap}).")
    return chunks

def create_vector_store(chunks, persist_directory="db/vector_store"):
    """Embed chunks and persist them into a Chroma vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created with {len(chunks)} chunks and persisted to '{persist_directory}'.")
    return vector_store

def main():
    directory_path = "data"
    persist_directory = "db/vector_store"

    # Step 1: Load raw documents from the data directory
    documents = load_documents_from_directory(directory_path)

    # Step 2: Split documents into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)

    # Step 3: Embed chunks and store in Chroma vector store
    vector_store = create_vector_store(chunks, persist_directory=persist_directory)

    print("\nIngestion pipeline complete! Knowledge base is ready.")

if __name__ == "__main__":
    main()