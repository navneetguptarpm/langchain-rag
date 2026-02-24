from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

def load_vector_store(persist_directory="db/vector_store"):
    # Load the Chroma vector store from the specified directory
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

    # Show a message indicating that the vector store has been loaded
    print(f"Loaded vector store from '{persist_directory}' with {vector_store._collection.count()} vectors.")

    return vector_store # Return the loaded Chroma vector store

def answer_query(query, relevant_docs):
    combined_input = (
        f"Based on the following retrieved documents, answer the question: '{query}'\n\n"
        + "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])
        + "\n\nPlease provide a concise and accurate answer based on the information from the retrieved documents. "
        "If the information is not sufficient to answer the question, please indicate that as well."
    ) # Combine the query and retrieved documents into a single input for the LLM

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2) # Initialize the language model (LLM) with the specified model and temperature
    response = llm.invoke([SystemMessage(content="You are a helpful assistant that provides concise and accurate answers based on the provided documents."), HumanMessage(content=combined_input)]) # Generate a response from the LLM based on the combined input of the query and retrieved documents
    
    return response.content # Return the generated answer from the LLM


def main():
    vector_store = load_vector_store() # Load the vector store from the specified directory

    query = "What percentage of the discrete GPU market did Nvidia hold as of Q1 2025?"  # Replace with your query
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve the top 3 most relevant chunks from the vector store based on the query

    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"score_threshold": 0.3, "k": 5}
    # )

    relevant_docs = retriever.invoke(query) # Print the retrieved relevant documents based on the query
    
    # print(f"Retrieved {len(relevant_docs)} relevant documents for the query: '{query}'")
    # for i, doc in enumerate(relevant_docs):
    #     print(f"Document {i+1}:")
    #     print(f'Content Length: {len(doc.page_content)} characters')  # Print the length of the content
    #     print(f"Content: {doc.page_content}")  # Print the content of the document
    #     print(f"Metadata: {doc.metadata}")  # Print the metadata
    #     print("-" * 40)  # Separator between documents

    answer = answer_query(query, relevant_docs) # Generate an answer to the query based on the retrieved relevant documents
    print(f"Query: {query}\nAnswer: {answer}") # Print the query and the generated answer
    
if __name__ == "__main__":
    main()