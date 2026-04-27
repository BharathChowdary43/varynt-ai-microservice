from pydantic import BaseModel

class MatchRequest(BaseModel):
    query_text: str
    n_results: int = 1

class MatchResponse(BaseModel):
    match_found: str
    similarity_score: float