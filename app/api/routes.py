from fastapi import APIRouter, HTTPException
from app.models.schemas import MatchRequest, MatchResponse
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/match_lead", response_model=MatchResponse)
async def match_lead_endpoint(request: MatchRequest):
    try:
        match, score = ai_service.get_best_match(
            query=request.query_text,
            n_results=request.n_results
        )
        return MatchResponse(match_found=match, similarity_score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama/Chroma Error: {str(e)}")