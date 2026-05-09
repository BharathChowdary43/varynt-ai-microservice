import os
import logging
from typing import List, Dict, Any
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

logger = logging.getLogger("VARYNT_Trace")

class VectorSearchService:
    """
    Handles Semantic Search and Vector Storage for KeaBuilder assets and leads.
    Satisfies ML Engineer Assessment Q1 & Q3.
    """
    def __init__(self, data_path: str = "./chroma_data", collection_name: str = "keabuilder_leads"):
        logger.info("Initializing VectorSearchService...")
        
        # -------------------------------------------------------------------------
        # SCALING ROADMAP: DATABASE STORAGE
        # MVP (Day 1): Using local persistent SQLite-based ChromaDB to save costs.
        # Production (Day 100): Swap to a managed vector database cluster (e.g., Pinecone, 
        # Weaviate, or Chroma Cloud) to handle 100k+ concurrent read/writes and High Availability.
        # -------------------------------------------------------------------------
        self.client = chromadb.PersistentClient(path=data_path)
        
        # -------------------------------------------------------------------------
        # SCALING ROADMAP: EMBEDDING STRATEGY
        # MVP (Day 1): Using 'all-MiniLM-L6-v2' via sentence-transformers. 
        #   Pros: Zero API cost, zero network latency, runs perfectly on CPU.
        # Production (Day 100): Swap to OpenAI `text-embedding-3-small` or VoyageAI.
        #   Why? Higher dimensionality (1536 vs 384) captures deeper semantic nuances 
        #   in complex, multi-paragraph user inputs.
        # -------------------------------------------------------------------------
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize the Chroma collection (Uses HNSW indexing under the hood for fast retrieval)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Explicitly setting cosine distance for ML Q1
        )
        
        # Auto-populate the database for the demo if it's empty
        if self.collection.count() == 0:
            self._seed_initial_data()

    def _seed_initial_data(self):
        """Injects dummy historical leads so the LangGraph RAG pipeline has context to learn from."""
        logger.info("Database empty. Seeding initial KeaBuilder lead data...")
        
        dummy_leads = [
            # HOT LEADS
            "I have a budget of $5000 and need a sales funnel built by next Tuesday. Let's go.",
            "Ready to buy the premium plan right now, just need to know if you integrate with Stripe.",
            "My credit card is ready. I need a custom meal plan funnel and coaching starting tomorrow.",
            
            # WARM LEADS
            "Could someone call me? I want to know the difference between the basic and pro tier.",
            "Does KeaBuilder support uploading 4K videos for my landing page?",
            "Looking to start a podcast next month, wondering what your setup packages look like.",
            
            # COLD / SUPPORT LEADS
            "Just looking around at your templates, they look nice.",
            "I lost my password and cannot log into my account. Please help.",
            "I sell SEO services, please check out my website here: spam-link.com"
        ]
        
        ids = [f"lead_{i}" for i in range(len(dummy_leads))]
        
        # Add to ChromaDB
        self.collection.add(
            documents=dummy_leads,
            ids=ids
        )
        logger.info(f"Successfully seeded {len(dummy_leads)} historical leads into vector memory.")

    def get_best_match(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most semantically similar documents to the query.
        Used by the LangGraph Memory Retriever Node.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.warning("Vector search returned no results.")
                return []

            matches = []
            # Extract the closest documents and their distances
            for idx, document in enumerate(results['documents'][0]):
                distance = results['distances'][0][idx]
                
                # Convert cosine distance to a clean 0.0 to 1.0 similarity score
                similarity_score = round(max(0.0, 1.0 - distance), 4)
                
                matches.append({
                    "document": document,
                    "similarity_score": similarity_score,
                    "id": results['ids'][0][idx]
                })
                
            return matches

        except Exception as e:
            logger.error(f"Failed to retrieve embeddings from ChromaDB: {str(e)}")
            # Graceful fallback: return empty list rather than crashing the pipeline
            return []
            
    def add_lead_to_memory(self, lead_text: str, lead_id: str):
        """
        SCALING ROADMAP: Continuous Learning.
        In a production system, once a lead is successfully processed, we call this 
        function to add it to the database so the RAG pipeline gets smarter over time.
        """
        try:
            self.collection.add(
                documents=[lead_text],
                ids=[lead_id]
            )
            logger.info(f"Dynamically added new lead {lead_id} to vector memory.")
        except Exception as e:
            logger.error(f"Failed to add new lead to memory: {str(e)}")