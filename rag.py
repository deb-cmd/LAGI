from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.agent import Agent
from agno.models.ollama import Ollama

# Create embedder instance (optionally specify model)
embedder = SentenceTransformerEmbedder()

# Initialize ChromaDB with Sentence Transformer
pdf_knowledge_base = PDFKnowledgeBase(
    path="data",
    vector_db=ChromaDb(
        collection="docs",
        embedder=embedder,
        # persist_path="./chroma_data"  # Uncomment for persistent storage
    ),
    reader=PDFReader(chunk=True),
)

# Create agent with ChromaDB knowledge base
agent = Agent(
    model=Ollama(id = "qwen2.5:3b"),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
)

# Test the embeddings
sample_embedding = embedder.get_embedding("Sample text chunk")
print(f"Dimensions: {len(sample_embedding)}")

# Load documents and query
agent.knowledge.load(recreate=False)
agent.print_response("What is the state of open souce ai?")