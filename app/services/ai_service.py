import chromadb
import chromadb.utils.embedding_functions as embedding_functions

class LocalAIService:
    def __init__(self):
        # 1. Setup local storage for ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_data")
        
        # 2. Use the lightweight, open-source sentence-transformer model directly in Python
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. Create or load the collection
        self.collection = self.client.get_or_create_collection(
            name="keabuilder_leads",
            embedding_function=self.sentence_transformer_ef
        )
        
        # 4. Auto-populate if empty
        if self.collection.count() == 0:
            self._load_dataset()

    def _load_dataset(self):
        print("Initializing database with KeaBuilder dummy lead submissions...")
        
        # Dummy inputs representing raw form submissions from KeaBuilder leads
        dataset = [
            # HOT leads (High intent, budget, urgency)
            "I have a budget of $5000 and need a sales funnel built by next Tuesday. Let's go.",
            "Ready to buy the premium plan right now, just need to know if you integrate with Stripe.",
            "My credit card is ready. I need a custom meal plan and 1-on-1 coaching starting tomorrow.",
            "Looking to hire an agency immediately to scale our Facebook ads. Budget is $10k/month.",
            "I need to launch my e-commerce store this weekend. Please call me ASAP.",
            
            # WARM leads (Interested, asking questions)
            "Can you send me your pricing options for the fitness coaching program?",
            "I'm interested in your software, but do you offer a free trial first?",
            "How much does it usually cost to have someone design a landing page?",
            "Looking to start a podcast next month, wondering what your setup packages look like.",
            "Do you guys offer discounts for non-profits? Trying to figure out my budget.",
            
            # COLD leads (Just browsing, vague, or spam)
            "Just looking around at your templates, they look nice.",
            "What exactly is a sales funnel anyway?",
            "Hi, do you accept guest posts on your blog?",
            "Not sure what I need yet, just downloading the free PDF guide.",
            "I sell SEO services, please check out my website here: spam-link.com"
        ]
        
        ids = [f"lead_id_{i}" for i in range(len(dataset))]
        self.collection.add(documents=dataset, ids=ids)
        print("Lead database successfully populated!")

    def get_best_match(self, query: str, n_results: int = 1):
        # Search the database for the closest semantic match
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract the closest document
        best_match_text = results['documents'][0][0]
        distance = results['distances'][0][0]
        
        # Convert cosine distance to a clean 0-1 similarity score
        score = round(max(0.0, 1.0 - distance), 4)
        
        return best_match_text, score

# Instantiate the service so it is ready when the web server boots
ai_service = LocalAIService()