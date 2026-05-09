import asyncio
import traceback
from app.services.search_service import VectorSearchService
from app.services.lead_pipeline import ReflexionLeadPipeline

async def run_manual_test():
    print("--- 1. Booting Services ---")
    try:
        search_service = VectorSearchService()
        pipeline = ReflexionLeadPipeline(vector_store_service=search_service)
        
        test_lead = "I want to create a funnel for my gym with a budget of $5000"
        
        print(f"\n--- 2. Sending Lead to Pipeline ---")
        print(f"Lead Text: '{test_lead}'\n")
        
        # Run the pipeline directly, bypassing FastAPI
        result = await pipeline.process_lead(test_lead)
        
        print("\n--- 3. SUCCESS! Final Output ---")
        print(result)
        
    except Exception as e:
        print("\n================ FATAL CRASH DETECTED ================")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\n--- FULL TRACEBACK ---")
        traceback.print_exc()
        print("======================================================")

if __name__ == "__main__":
    asyncio.run(run_manual_test())