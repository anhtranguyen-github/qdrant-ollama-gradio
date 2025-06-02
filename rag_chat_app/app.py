import gradio as gr
from gradio import ChatMessage
import ollama
import time
import os
from uuid import uuid4
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import traceback


class DocumentRetrieverTool:
    def __init__(self, collection_name="rag-qdrant", embedding_model="nomic-embed-text"):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_size = 768  # nomic-embed-text uses 768 dimensions
        self.client = None
        self.vectorstore = None
        self.retriever_tool = None
        self.document_count = 0
        
        # Get configuration from environment variables
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        self.setup_retriever()
    
    def setup_retriever(self):
        """Initialize the retriever with sample documents"""
        try:
            print(f"Initializing retriever with model: {self.embedding_model}")
            print(f"Ollama URL: {self.ollama_base_url}")
            print(f"Qdrant: {self.qdrant_host}:{self.qdrant_port}")
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_base_url
            )
            
            # Test embedding first
            print("Testing embedding model...")
            test_embed = self.embeddings.embed_query("test")
            print(f"Embedding test successful. Vector size: {len(test_embed)}")
            
            # Create Qdrant client
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            print("Qdrant client created")
            
            # Create collection
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=len(test_embed),  # Use actual embedding size
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Collection '{self.collection_name}' created")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"Collection '{self.collection_name}' already exists")
                else:
                    raise e
            
            # Initialize vectorstore
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            print("VectorStore initialized")
            
            # Load initial documents - start with simple test documents
            self.load_test_documents()
            
            # Create retriever tool
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            self.retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and return relevant information from the document knowledge base.",
            )
            
            print(f"Document retriever initialized successfully with {self.document_count} documents")
            
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            traceback.print_exc()
    
    def load_test_documents(self):
        """Load simple test documents first to ensure the system works"""
        try:
            print("Loading test documents...")
            
            # Simple test documents
            test_docs = [
                Document(
                    page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                    metadata={"source": "test", "title": "ML Introduction"}
                ),
                Document(
                    page_content="Large language models are neural networks trained on vast amounts of text data to understand and generate human language.",
                    metadata={"source": "test", "title": "LLM Basics"}
                ),
                Document(
                    page_content="Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to provide more accurate and contextual responses.",
                    metadata={"source": "test", "title": "RAG Overview"}
                ),
                Document(
                    page_content="Vector databases store high-dimensional vectors and enable fast similarity search for applications like semantic search and recommendation systems.",
                    metadata={"source": "test", "title": "Vector Databases"}
                ),
                Document(
                    page_content="Docker containers provide a lightweight, portable way to package applications and their dependencies for consistent deployment across different environments.",
                    metadata={"source": "test", "title": "Docker Basics"}
                ),
                Document(
                    page_content="Qdrant is an open-source vector database that provides fast and scalable vector similarity search for machine learning applications.",
                    metadata={"source": "test", "title": "Qdrant Database"}
                )
            ]
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            doc_splits = text_splitter.split_documents(test_docs)
            
            if doc_splits:
                uuids = [str(uuid4()) for _ in range(len(doc_splits))]
                
                # Add to vectorstore
                self.vectorstore.add_documents(documents=doc_splits, ids=uuids)
                self.document_count = len(doc_splits)
                print(f"Added {len(doc_splits)} test document chunks to the knowledge base")
                
                # Test retrieval
                test_query = "What is machine learning?"
                retriever = self.vectorstore.as_retriever()
                results = retriever.invoke(test_query)
                print(f"Test retrieval found {len(results)} documents")
                
            # Try to load web documents (optional)
            self.load_web_documents()
                
        except Exception as e:
            print(f"Error loading test documents: {e}")
            traceback.print_exc()
    
    def load_web_documents(self):
        """Try to load web documents (optional)"""
        try:
            print("Attempting to load web documents...")
            urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            ]
            
            web_docs = []
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    web_docs.extend(docs)
                    print(f"Loaded: {url}")
                except Exception as e:
                    print(f"Failed to load {url}: {e}")
                    continue
            
            if web_docs:
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                doc_splits = text_splitter.split_documents(web_docs)
                uuids = [str(uuid4()) for _ in range(len(doc_splits))]
                
                # Add to vectorstore
                self.vectorstore.add_documents(documents=doc_splits, ids=uuids)
                self.document_count += len(doc_splits)
                print(f"Added {len(doc_splits)} web document chunks to the knowledge base")
            else:
                print("No web documents loaded - using test documents only")
                
        except Exception as e:
            print(f"Error loading web documents (will continue with test docs): {e}")
    
    def add_text_document(self, text, metadata=None):
        """Add a text document to the knowledge base"""
        try:
            if not text.strip():
                return False, "Empty text provided"
            
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # Split if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            doc_splits = text_splitter.split_documents([doc])
            uuids = [str(uuid4()) for _ in range(len(doc_splits))]
            
            # Add to vectorstore
            self.vectorstore.add_documents(documents=doc_splits, ids=uuids)
            self.document_count += len(doc_splits)
            return True, f"Added {len(doc_splits)} chunks to knowledge base"
            
        except Exception as e:
            return False, f"Error adding document: {str(e)}"
    
    def search_documents(self, query):
        """Search documents using the retriever tool"""
        try:
            if self.retriever_tool is None:
                return []
            
            result = self.retriever_tool.invoke({"query": query})
            return result
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return f"Error during search: {str(e)}"


# Initialize document retriever
try:
    doc_retriever = DocumentRetrieverTool()
    retrieval_enabled = True
    print("Document retrieval system initialized successfully")
except Exception as e:
    print(f"Failed to initialize document retrieval: {e}")
    retrieval_enabled = False
    doc_retriever = None


def ollama_thinking_chat(message, history):
    start_time = time.time()
    
    # Get Ollama configuration
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = "qwen3:8b"  # Updated model name

    
    # Configure Ollama client
    client = ollama.Client(host=ollama_base_url)
    
    # First show thinking step
    thinking_response = ChatMessage(
        content="",
        metadata={"title": "_Thinking_ step-by-step", "id": 0, "status": "pending"}
    )
    yield thinking_response
    
    # Document retrieval step
    retrieval_response = ChatMessage(
        content="",
        metadata={"title": "_Tool_ retrieve document", "id": 1, "status": "pending"}
    )
    yield [thinking_response, retrieval_response]
    
    # Perform document retrieval
    retrieved_content = ""
    retrieval_content = ""
    
    if retrieval_enabled and doc_retriever:
        try:
            retrieval_content = "Searching knowledge base...\n"
            retrieval_response.content = retrieval_content
            yield [thinking_response, retrieval_response]
            
            # Use the retriever tool
            search_result = doc_retriever.search_documents(message)
            
            if isinstance(search_result, str) and search_result:
                # Format the retrieved content
                retrieved_content = search_result
                retrieval_content = f"Retrieved relevant information:\n\n{search_result[:500]}..."
                if len(search_result) > 500:
                    retrieval_content += f"\n\n[Total content length: {len(search_result)} characters]"
            else:
                retrieval_content = "No relevant documents found in the knowledge base."
                
            retrieval_response.content = retrieval_content
        except Exception as e:
            retrieval_response.content = f"Error during document retrieval: {str(e)}"
    else:
        retrieval_response.content = "Document retrieval system not available."
    
    # Mark retrieval as done
    retrieval_response.metadata["status"] = "done"
    yield [thinking_response, retrieval_response]

    # Build chat messages with context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Add retrieved documents to context if available
    if retrieved_content:
        context = f"Here is relevant information from the knowledge base that might help answer the user's question:\n\n{retrieved_content}\n\nPlease use this information to help answer the user's question when relevant."
        messages.append({"role": "system", "content": context})
    
    # Add conversation history
    for h in history:
        if h["role"] == "user":
            messages.append({"role": "user", "content": h["content"]})
        elif h["role"] == "assistant":
            messages.append({"role": "assistant", "content": h["content"]})
    
    messages.append({"role": "user", "content": message})

    # Start streaming from Ollama
    try:
        completion = client.chat(
            model=chat_model,
            messages=messages,
            stream=True
        )

        thinking = ""
        final_response = ""
        is_thinking = False

        for chunk in completion:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']

                # Handle <think> tags for thought reasoning
                if "<think>" in content:
                    is_thinking = True
                    content = content.split("<think>", 1)[-1]

                if "</think>" in content:
                    is_thinking = False
                    content = content.split("</think>", 1)[0]
                    thinking += content
                    thinking_response.content = "- " + thinking.strip().replace("\n", "\n- ")
                    yield [thinking_response, retrieval_response]
                    continue

                if is_thinking:
                    thinking += content
                    thinking_response.content = "- " + thinking.strip().replace("\n", "\n- ")
                    yield [thinking_response, retrieval_response]
                else:
                    final_response += content

        # Finish thinking message
        thinking_response.metadata["status"] = "done"
        thinking_response.metadata["duration"] = time.time() - start_time
        yield [thinking_response, retrieval_response]

        # Send final assistant message
        yield [
            thinking_response,
            retrieval_response,
            ChatMessage(
                content="Based on my thoughts and retrieved information, my response is:\n\n" + final_response.strip()
            )
        ]
        
    except Exception as e:
        # Handle Ollama errors
        thinking_response.metadata["status"] = "error"
        yield [
            thinking_response,
            retrieval_response,
            ChatMessage(
                content=f"Error generating response: {str(e)}"
            )
        ]


def add_document_to_db(text, title="", source=""):
    """Function to add documents to the database"""
    if not retrieval_enabled or not doc_retriever:
        return "Document retrieval system not available."
    
    if not text.strip():
        return "Please enter some text to add to the database."
    
    try:
        metadata = {}
        if title.strip():
            metadata["title"] = title.strip()
        if source.strip():
            metadata["source"] = source.strip()
        
        success, message = doc_retriever.add_text_document(text, metadata)
        return message
    except Exception as e:
        return f"Error adding document: {str(e)}"


# Create the main chat interface
chat_interface = gr.ChatInterface(
    ollama_thinking_chat,
    title="Thinking LLM Chat Interface with Document Retrieval 🤔📚",
    type="messages",
)

# Create the full application
with gr.Blocks(title="RAG Chat with Thinking") as app:
    gr.Markdown("# Thinking LLM Chat with RAG Document Retrieval")
    gr.Markdown("This system combines thinking steps with document retrieval using LangChain and Qdrant.")
    
    with gr.Tab("💬 Chat"):
        chat_interface.render()
    
    with gr.Tab("📚 Document Management"):
        gr.Markdown("## Add Documents to Knowledge Base")
        gr.Markdown("Add your own documents to enhance the AI's knowledge base.")
        
        with gr.Row():
            with gr.Column():
                doc_title = gr.Textbox(
                    label="Document Title (optional)",
                    placeholder="Enter document title...",
                    lines=1
                )
                doc_source = gr.Textbox(
                    label="Source (optional)",
                    placeholder="Enter source URL or reference...",
                    lines=1
                )
                doc_text = gr.Textbox(
                    label="Document Text",
                    placeholder="Enter the document content here...",
                    lines=15
                )
                add_btn = gr.Button("Add Document", variant="primary", size="lg")
                
            with gr.Column():
                result_text = gr.Textbox(
                    label="Result",
                    interactive=False,
                    lines=5
                )
        
        add_btn.click(
            fn=add_document_to_db,
            inputs=[doc_text, doc_title, doc_source],
            outputs=[result_text]
        )
        
        gr.Markdown("""
        ### 📋 Current Knowledge Base
        The system starts with documents from Lilian Weng's blog posts about:
        - AI Agents
        - Prompt Engineering 
        - Adversarial Attacks on LLMs
        
        ### 🔧 Setup Requirements:
        ```bash
        # Install required packages
        pip install langchain-community langchain-qdrant langchain-ollama qdrant-client langchain-text-splitters beautifulsoup4 lxml
        
        # Pull the embedding model in Ollama
        ollama pull nomic-embed-text:latest
        
        # Make sure Ollama is running
        ollama serve
        ```
        
        ### 🐛 Troubleshooting:
        - **No documents found**: Check console logs for initialization errors
        - **Embedding errors**: Ensure `nomic-embed-text:latest` is pulled in Ollama
        - **Connection issues**: Verify Ollama is running on localhost:11434
        
        ### 💡 Usage Tips:
        1. **Add Documents**: Use the form above to expand the knowledge base
        2. **Ask Questions**: Try questions about ML, AI, or RAG to test the system
        3. **Check Logs**: Look at console output for debugging information
        
        ### ⚙️ Configuration:
        - **Embedding Model**: nomic-embed-text:latest (768 dimensions)
        - **Vector Store**: Qdrant (in-memory)
        - **Chat Model**: qwen3:8b (via Ollama)
        - **Test Documents**: Includes basic AI/ML concepts for testing
        """)

if __name__ == "__main__":
    # Get server configuration from environment
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    print(f"Starting Gradio app on {server_name}:{server_port}")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False
    ) 