from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging

# Import our two core services
from app.services.search_service import VectorSearchService
from app.services.lead_pipeline import ReflexionLeadPipeline

# Set up logging to match our pipeline
logger = logging.getLogger("VARYNT_Trace")

app = FastAPI(
    title="VARYNT AI Microservice",
    description="Core backend handling Semantic Search and Multi-Agent Lead Qualification",
    version="1.0.0"
)

# -------------------------------------------------------------------------
# 1. Initialize Services (Loads ChromaDB into memory on startup)
# -------------------------------------------------------------------------
print("Booting VARYNT AI Services...")
search_service = VectorSearchService()
lead_pipeline = ReflexionLeadPipeline(vector_store_service=search_service)

# -------------------------------------------------------------------------
# 2. API Request Schemas
# -------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="The text to search for.")
    top_k: int = Field(default=3, description="Number of results to return.")

class LeadRequest(BaseModel):
    # Enforce a minimum length at the API level before it even hits the Agent
    text: str = Field(..., min_length=10, description="The raw lead text from the CRM.")

# -------------------------------------------------------------------------
# 3. The API Endpoints
# -------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "varynt-ai-microservice"}

@app.post("/search", tags=["ML Track: Vector Search"])
async def semantic_search(request: SearchRequest):
    """Solves ML Engineer Q1: Retrieves most similar results from vector memory."""
    try:
        results = search_service.get_best_match(request.query, top_k=request.top_k)
        return {"matches": results}
    except Exception as e:
        logger.error(f"Search endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal vector search failure.")

@app.post("/process_lead", tags=["AI Track: Multi-Agent Engine"])
async def process_new_lead(request: LeadRequest):
    """Solves AI Engineer Q2: Evaluates, classifies, and drafts responses via LangGraph."""
    try:
        # Await the asynchronous LangGraph pipeline
        result = await lead_pipeline.process_lead(request.text)
        return result
    except ValueError as ve:
        # Catch the Evaluator Agent's explicit rejections (e.g., spam)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lead processing failed: {str(e)}")
        raise HTTPException(status_code=502, detail="AI generation pipeline failed or timed out.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)