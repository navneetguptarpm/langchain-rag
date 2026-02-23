import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

# This script demonstrates how to load documents from a directory using LangChain's DirectoryLoader and TextLoader.
def load_documents_from_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
    
    # Use DirectoryLoader to load all .txt files in the directory
    loader = DirectoryLoader(path=directory_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Check if any documents were loaded
    if not documents:
        raise ValueError(f"No text files found in directory '{directory_path}'.")
    
    # Show the number of documents loaded
    print(f"Loaded {len(documents)} documents from '{directory_path}'.")

    return documents # Return the loaded documents as a list of Document objects (with 'page_content' and 'metadata' attributes)

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    # Use CharacterTextSplitter to split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # Show the number of chunks created
    print(f"Split documents into {len(chunks)} chunks (chunk size: {chunk_size} characters).")

    return chunks # Return the list of split Document objects

def create_vector_store(chunks, persist_directory="db/vector_store"):
    # Create OpenAI embeddings for the chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Chroma vector store from the chunks and persist it to disk
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})

    # Show a message indicating that the vector store has been created and persisted
    print(f"Created vector store with {len(chunks)} chunks and persisted to '{persist_directory}'.")

    return vector_store # Return the created Chroma vector store

def main():
    directory_path = "data"  # Replace with your directory path
    documents = load_documents_from_directory(directory_path) # Load documents from the specified directory

    # Print the first document object to see its structure (it should have 'page_content' and 'metadata' attributes)
    # if documents:
    #     for i, doc in enumerate(documents):
    #         print(f"Document {i+1}:")
    #         print(f'Content Length: {len(doc.page_content)} characters')  # Print the length of the content
    #         print(f"Content: {doc.page_content[:100]}...")  # Print the first 100 characters of the content
    #         print(f"Metadata: {doc.metadata}")  # Print the metadata
    #         print("-" * 40)  # Separator between documents
    # else:
    #     print("No documents were loaded.")

    chunks = split_documents(documents) # Split the loaded documents into smaller chunks

    # for i, chunk in enumerate(chunks[:5]):  # Print the first 5 chunks to see their structure
    #     print(f"Chunk {i+1}:")
    #     print(f'Content Length: {len(chunk.page_content)} characters')  # Print the length of the chunk content
    #     print(f"Content: {chunk.page_content}")  # Print the first 100 characters of the chunk content
    #     print(f"Metadata: {chunk.metadata}")  # Print the metadata of the chunk
    #     print("-" * 40)  # Separator between chunks

    vector_store = create_vector_store(chunks) # Create a Chroma vector store from the chunks and persist it to disk

if __name__ == "__main__":
    main()