from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

def load_vector_store(persist_directory="db/vector_store"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
    print(f"Loaded vector store from '{persist_directory}' with {vector_store._collection.count()} vectors.")
    return vector_store

def retrieve_docs(query, retriever):
    """Retrieve the most relevant documents for the given query."""
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def answer_query(query, relevant_docs, llm):
    """Generate an answer using the query and retrieved documents."""
    combined_input = (
        f"Based on the following retrieved documents, answer the question: '{query}'\n\n"
        + "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])
        + "\n\nPlease provide a concise and accurate answer based on the information from the retrieved documents. "
        "If the information is not sufficient to answer the question, please indicate that as well."
    )

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that provides concise and accurate answers based on the provided documents."),
        HumanMessage(content=combined_input)
    ])

    return response.content

def main():
    vector_store = load_vector_store()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    print("Basic RAG Chatbot (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye! Have a great day! 👋")
            break
        if not query:
            continue

        # Step 1: Retrieve relevant documents
        relevant_docs = retrieve_docs(query, retriever)

        # Step 2: Generate answer from retrieved docs
        answer = answer_query(query, relevant_docs, llm)
        print(f"Assistant: {answer}\n")

if __name__ == "__main__":
    main()