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

def contextualize_query(query, chat_history, llm):
    """Reformulate the query to be standalone based on chat history."""
    if not chat_history:
        return query  # No history, return query as-is

    history_text = "\n".join(
        [f"User: {msg['query']}\nAssistant: {msg['answer']}" for msg in chat_history]
    )

    contextualize_prompt = (
        f"Given the following conversation history:\n{history_text}\n\n"
        f"And the follow-up question: '{query}'\n\n"
        "Rewrite the follow-up question as a standalone question that captures all necessary context from the conversation history. "
        "Return ONLY the rewritten question, nothing else."
    )

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that reformulates follow-up questions into standalone questions."),
        HumanMessage(content=contextualize_prompt)
    ])

    return response.content.strip()

def answer_query(query, relevant_docs, chat_history, llm):
    """Generate an answer using retrieved docs and chat history."""
    history_text = ""
    if chat_history:
        history_text = "Conversation history:\n" + "\n".join(
            [f"User: {msg['query']}\nAssistant: {msg['answer']}" for msg in chat_history]
        ) + "\n\n"

    combined_input = (
        f"{history_text}"
        f"Based on the following retrieved documents, answer the question: '{query}'\n\n"
        + "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])
        + "\n\nPlease provide a concise and accurate answer based on the information from the retrieved documents. "
        "If the information is not sufficient to answer the question, please indicate that as well."
    )

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that provides concise and accurate answers based on the provided documents and conversation history."),
        HumanMessage(content=combined_input)
    ])

    return response.content

def main():
    vector_store = load_vector_store()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    chat_history = []  # Stores list of {"query": ..., "answer": ...}

    print("History-Aware RAG Chatbot (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye! Have a great day! 👋")
            break
        if not query:
            continue

        # Step 1: Reformulate query using chat history for better retrieval
        standalone_query = contextualize_query(query, chat_history, llm)
        # print(f"[Standalone Query]: {standalone_query}")

        # Step 2: Retrieve relevant docs using the standalone query
        relevant_docs = retriever.invoke(standalone_query)

        # Step 3: Generate answer with original query + history + docs
        answer = answer_query(query, relevant_docs, chat_history, llm)
        print(f"Assistant: {answer}\n")

        # Step 4: Update chat history
        chat_history.append({"query": query, "answer": answer})

if __name__ == "__main__":
    main()