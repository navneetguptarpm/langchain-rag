import os
from dotenv import load_dotenv

# LangChain Document Loaders
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# 1 & 2. Standard Text Splitters
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 3. Document Specific Splitter (Markdown)
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 4. Semantic Splitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 5. Agentic Splitter (LLM-based Structured Output)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# --- HELPER FUNCTIONS ---

def load_documents_from_directory(directory_path="data"):
    """Loads text documents from a directory."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path) # Auto-create directory for testing
        print(f"Created directory '{directory_path}'. Please add a .txt file and run again.")
        return []
    
    loader = DirectoryLoader(path=directory_path, glob="*.txt", loader_cls=TextLoader)
    return loader.load()

def print_chunks(chunks, strategy_name):
    """Utility function to print chunks clearly."""
    print(f"{'=' * (len(strategy_name) + 2)}\n{strategy_name}\n{'=' * (len(strategy_name) + 2)}")
    for i, chunk in enumerate(chunks):
        # Handle different chunk object types (Document vs string)
        content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
        print(f"Chunk {i+1} ({len(content)} chars):\n{content}\n{'-' * 50}")


# --- STRATEGY 1: CHARACTER SPLITTING ---

def strategy_1_character(documents):
    """Splits blindly by a specific character (\n\n by default)."""
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=20
    )
    return text_splitter.split_documents(documents)


# --- STRATEGY 2: RECURSIVE CHARACTER SPLITTING ---

def strategy_2_recursive(documents):
    """Best practice default. Falls back through separators (\n\n -> \n -> space) to keep sizes."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=20,
        # Default separators are ["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# --- STRATEGY 3: DOCUMENT SPECIFIC SPLITTING (MARKDOWN) ---

def strategy_3_markdown(markdown_text):
    """Splits based on markdown headers. 
    Best Practice: We pipe the output into a Recursive splitter to enforce the 100 char limit!"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # 1. Split by structure
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    # 2. Enforce chunk size constraint
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return recursive_splitter.split_documents(md_header_splits)


# --- STRATEGY 4: SEMANTIC SPLITTING ---

def strategy_4_semantic(documents):
    """Splits by meaning. 
    Note: chunk_size=100 is NOT used here. It splits when topics change."""
    
    # Requires an embedding model to compare sentence similarities
    embeddings = OpenAIEmbeddings()
    
    semantic_chunker = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" # Splits when similarity drops below a certain percentile
    )
    return semantic_chunker.split_documents(documents)


# --- STRATEGY 5: AGENTIC (PROPOSITIONAL) SPLITTING ---

# Define the structured output we want the LLM to return
class Propositions(BaseModel):
    sentences: list[str] = Field(
        description="A list of standalone, decontextualized propositions or facts extracted from the text."
    )

def strategy_5_agentic(text):
    """Uses an LLM to rewrite text into standalone facts.
    Note: chunk_size=100 is NOT used. Chunk size depends on the idea length."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Propositions)
    
    prompt = f"""
    Decompose the following text into clear, simple, and standalone propositions.
    Each proposition should make perfect sense on its own without needing the surrounding text.
    Replace pronouns with specific names where necessary.
    
    Text: {text}
    """
    
    result = structured_llm.invoke(prompt)
    return result.sentences


# --- MAIN EXECUTION ---

def main():
    docs = load_documents_from_directory("data")
    
    if not docs:
        return

    # For testing document specific and agentic strategies which take raw text
    sample_text = docs[0].page_content 

    # 1. Character
    print_chunks(strategy_1_character(docs), "1. Character Chunking")

    # 2. Recursive
    # print_chunks(strategy_2_recursive(docs), "2. Recursive Chunking")

    # 3. Markdown (Assumes sample text has markdown headers)
    # print_chunks(strategy_3_markdown(sample_text), "3. Markdown Chunking")

    # The following require OpenAI API keys. Uncomment to test if you have an active key in your .env
    # 4. Semantic
    # print_chunks(strategy_4_semantic(docs), "4. Semantic Chunking")

    # 5. Agentic
    # agentic_chunks = strategy_5_agentic(sample_text)
    # print_chunks(agentic_chunks, "5. Agentic Chunking")

if __name__ == "__main__":
    main()