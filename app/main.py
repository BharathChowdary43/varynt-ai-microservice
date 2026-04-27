from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="VARYNT AI Microservice",
    description="Local AI Backend for KeaBuilder Tasks using ChromaDB",
    version="1.0.0"
)

# Attach our API endpoints
app.include_router(router, prefix="/api")

@app.get("/")
async def health_check():
    return {"status": "Active", "message": "API is running. Send POST to /api/match_lead"}