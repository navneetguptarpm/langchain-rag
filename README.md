

# Modern LLM and RAG Stack Architecture

This document outlines the core dependencies used in our Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) applications. 

Recently, the LangChain ecosystem underwent a major architectural shift to become more modular. Instead of a single monolithic package, integrations and core logic are now split into specialized packages. This improves dependency management, security, and integration updates.



Below is the breakdown of the packages powering our stack, their responsibilities, and how they are typically used.

---

## Environment Management

### `python-dotenv`
* **Role:** Environment variable manager.
* **Responsibilities:** Reads key-value pairs from a `.env` file and adds them to the Python application's environment variables (`os.environ`).
* **Typical Usage:** LLM and database applications require secure handling of API keys (like `OPENAI_API_KEY`) and connection strings. `python-dotenv` ensures these secrets are loaded into the application without hardcoding them into the source code.

**Example:**
```python
from dotenv import load_dotenv

# Loads variables from .env into the environment
load_dotenv() 

```

---

## The LangChain Ecosystem

### 1. `langchain`

* **Role:** The orchestration framework.
* **Responsibilities:** Contains the application-level logic that ties everything together. It houses the higher-level abstractions like Chains (chains of operations), Agents (LLMs deciding which tools to use), and overarching retrieval strategies. It builds upon `langchain-core` but does *not* contain specific third-party integrations.
* **Typical Usage:** Used to create conversational retrieval chains or reasoning agents.

**Example:**

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Orchestrating the LLM and the Retriever
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

```

### 2. `langchain-community`

* **Role:** The general integration hub.
* **Responsibilities:** Houses thousands of third-party integrations (document loaders, lesser-used vector stores, web search tools, etc.) that do not yet have their own dedicated "partner package." It is maintained by the open-source community.
* **Typical Usage:** Used when you need to load data from specific sources (e.g., PDF loaders, Wikipedia loaders) or connect to tools that aren't major, standalone partners.

**Example:**

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()

```

### 3. `langchain_text_splitters`

* **Role:** Document chunking utility.
* **Responsibilities:** Provides algorithms to split large documents into smaller, semantically meaningful chunks. This is a crucial step in RAG, as LLMs have context window limits, and vector databases perform better with targeted text sizes.
* **Typical Usage:** Used immediately after loading documents and before embedding them into a vector store.

**Example:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

```

---

## Partner Packages (Dedicated Integrations)

To prevent the core LangChain framework from becoming bloated and to allow updates to track closely with official SDKs, major providers now have their own dedicated "partner packages."

### 4. `langchain_openai`

* **Role:** The intelligence and embedding engine.
* **Responsibilities:** Provides seamless, fully up-to-date integration with OpenAI's API. It contains the classes for OpenAI's chat models (`ChatOpenAI`) and embedding models (`OpenAIEmbeddings`).
* **Typical Usage:** Used to initialize the core LLM that generates responses and the embedding model that converts text into vector numbers for search.

**Example:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize the embedding model for the vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the LLM for generation
llm = ChatOpenAI(model="gpt-4o", temperature=0)

```

### 5. `langchain_chroma`

* **Role:** The Vector Database.
* **Responsibilities:** The dedicated integration for ChromaDB, a popular open-source vector database. It handles the storage of the document embeddings and performs the mathematical similarity searches (cosine similarity) to retrieve the most relevant chunks based on a user's query.
* **Typical Usage:** Used to ingest the chunks created by `langchain_text_splitters`, embed them using `langchain_openai`, and act as the `Retriever` in the final LangChain orchestration.

**Example:**

```python
from langchain_chroma import Chroma

# Store chunks in the vector database
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Use it as a retriever in a RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

```