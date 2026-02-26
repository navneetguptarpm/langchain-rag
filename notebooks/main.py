import os
import json
import logging
from typing import List, Dict, Any

# Disable ChromaDB telemetry logs before importing Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Langchain components
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# UPDATE: Using the new langchain-chroma package to fix the deprecation warning
from langchain_chroma import Chroma

# Load environment variables (e.g., OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MultimodalRAGPipeline:
    def __init__(self, persist_directory: str = "dbv2/chroma_db", model_name: str = "gpt-4o"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.0)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # ==========================================
    # STEP 1: PARSING & CHUNKING
    # ==========================================
    def partition_document(self, file_path: str) -> List[Any]:
        logger.info(f"Partitioning document: {file_path}")
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        return elements

    def create_chunks(self, elements: List[Any]) -> List[Any]:
        logger.info("Creating chunks by title...")
        chunks = chunk_by_title(
            elements=elements,
            max_characters=3000,
            new_after_n_chars=2500,
            combine_text_under_n_chars=500
        )
        return chunks

    # ==========================================
    # STEP 2: CONTENT ANALYSIS & AI SUMMARY
    # ==========================================
    def _extract_chunk_content(self, chunk: Any) -> Dict[str, Any]:
        content_data = {'text': chunk.text, 'tables': [], 'images': [], 'types': {'text'}}
        
        if not (hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
            content_data['types'] = list(content_data['types'])
            return content_data
            
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == 'Table':
                content_data['types'].add('table')
                content_data['tables'].append(getattr(element.metadata, 'text_as_html', element.text))
            elif element_type == 'Image' and hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                content_data['types'].add('image')
                content_data['images'].append(element.metadata.image_base64)
        
        content_data['types'] = list(content_data['types'])
        return content_data

    def _generate_content_summary(self, text: str, tables: List[str], images: List[str]) -> str:
        prompt_text = f"You are creating a searchable description for document content retrieval.\n\nTEXT CONTENT:\n{text}\n\n"
        for i, table in enumerate(tables, 1):
            prompt_text += f"TABLE {i}:\n{table}\n\n"
            
        prompt_text += (
            "YOUR TASK: Generate a comprehensive, searchable description covering:\n"
            "1. Key facts, numbers, and data points\n"
            "2. Main topics and alternative search terms\n"
            "3. Visual content analysis from any images provided\n"
            "SEARCHABLE DESCRIPTION:"
        )

        message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for img in images:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        try:
            response = self.llm.invoke([HumanMessage(content=message_content)])
            return response.content
        except Exception as e:
            logger.error(f"AI summary failed: {e}")
            return text[:500] + "... [Summary Failed]"

    def process_document_chunks(self, chunks: List[Any]) -> List[Document]:
        logger.info(f"Processing {len(chunks)} chunks for AI summaries...")
        langchain_docs = []
        
        for i, chunk in enumerate(chunks, 1):
            content_data = self._extract_chunk_content(chunk)
            
            if content_data['tables'] or content_data['images']:
                enhanced_content = self._generate_content_summary(
                    content_data['text'], content_data['tables'], content_data['images']
                )
            else:
                enhanced_content = content_data['text']
            
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images']
                    })
                }
            )
            langchain_docs.append(doc)
            
        return langchain_docs

    # ==========================================
    # STEP 3: VECTOR STORE & EXPORT
    # ==========================================
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        logger.info("Building ChromaDB vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vectorstore

    # ==========================================
    # STEP 4: RETRIEVAL & GENERATION
    # ==========================================
    def generate_final_answer(self, db: Chroma, query: str, k: int = 3) -> str:
        retriever = db.as_retriever(search_kwargs={"k": k})
        chunks = retriever.invoke(query)
        
        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT:\n"
        message_content: List[Dict[str, Any]] = []
        
        for i, chunk in enumerate(chunks, 1):
            prompt_text += f"--- Document {i} ---\n"
            try:
                original_data = json.loads(chunk.metadata.get("original_content", "{}"))
                prompt_text += f"TEXT:\n{original_data.get('raw_text', '')}\n\n"
                
                for j, table in enumerate(original_data.get("tables_html", []), 1):
                    prompt_text += f"TABLE {j}:\n{table}\n\n"
                    
                for img in original_data.get("images_base64", []):
                    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                    
            except json.JSONDecodeError:
                pass
            prompt_text += "\n"
        
        prompt_text += "ANSWER:"
        message_content.insert(0, {"type": "text", "text": prompt_text})
        
        response = self.llm.invoke([HumanMessage(content=message_content)])
        return response.content

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    PDF_PATH = "../data/NIPS-2017.pdf"
    
    # Initialize Pipeline
    pipeline = MultimodalRAGPipeline()
    
    # Check if vector DB already exists
    db_exists = os.path.exists(pipeline.persist_directory) and bool(os.listdir(pipeline.persist_directory))
    
    rebuild_db = False
    if db_exists:
        # Prompt user with default [N]
        user_choice = input(f"Existing database found at '{pipeline.persist_directory}'. Rebuild it? [y/N]: ").strip().lower()
        if user_choice in ['y', 'yes']:
            rebuild_db = True
    else:
        rebuild_db = True
        logger.info(f"No existing database found at '{pipeline.persist_directory}'. Starting initial build.")

    # Temporarily mute logging so the DB load is completely silent if choosing 'No'
    if not rebuild_db:
        logging.getLogger().setLevel(logging.ERROR)

    # Execute build or load
    if rebuild_db:
        raw_elements = pipeline.partition_document(PDF_PATH)
        raw_chunks = pipeline.create_chunks(raw_elements)
        processed_docs = pipeline.process_document_chunks(raw_chunks)
        vector_db = pipeline.create_vector_store(processed_docs)
    else:
        vector_db = Chroma(
            persist_directory=pipeline.persist_directory,
            embedding_function=pipeline.embedding_model
        )

    # Clean UI for Q&A Loop
    ui_width = 50
    print("\n" + "=" * ui_width)
    print("Multi-Modal RAG Agent!".center(ui_width))
    print("=" * ui_width)

    # Ensure all logging is disabled to keep the terminal output clean during Q&A
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Interactive Query Loop
    while True:
        try:
            # Get input from the user
            user_query = input("\nQuery: ").strip()
            
            # Check for exit command
            if user_query.lower() in ['exit', 'quit']:
                print("\nShutting down the agent. Goodbye! 👋")
                break
            
            # Skip if the user just pressed Enter without typing anything
            if not user_query:
                continue
                
            # Generate the answer using the pipeline
            answer = pipeline.generate_final_answer(vector_db, user_query)
            
            # Display the result cleanly
            print(f"Answer:\n{answer}")
            print("-" * ui_width)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nProcess interrupted by user. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n[Error] An error occurred while processing the query: {e}")