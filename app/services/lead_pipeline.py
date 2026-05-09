import logging
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama

# -------------------------------------------------------------------------
# SCALING ROADMAP: LLM INFRASTRUCTURE
# MVP: Using local Ollama with JSON mode and robust dict extraction.
# Production: Swap to langchain_openai and utilize native function calling.
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("varynt_agent_trace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VARYNT_Trace")

# -------------------------------------------------------------------------
# 1. Schemas (Strict JSON enforcement)
# -------------------------------------------------------------------------
class EvaluatorDecision(BaseModel):
    is_valid_sales_lead: bool
    reason: str

class DraftOutput(BaseModel):
    classification: str
    extracted_entity: str
    drafted_email: str

class ValidatorDecision(BaseModel):
    is_approved: bool
    feedback: str

# -------------------------------------------------------------------------
# 2. Graph State
# -------------------------------------------------------------------------
class LeadState(TypedDict):
    lead_text: str
    rag_context: str
    is_valid_lead: bool
    rejection_reason: str
    classification: str
    drafted_email: str
    feedback: str
    retry_count: int

# -------------------------------------------------------------------------
# 3. The Agents (LangGraph Nodes)
# -------------------------------------------------------------------------
class ReflexionLeadPipeline:
    def __init__(self, vector_store_service):
        logger.info("Initializing ReflexionLeadPipeline (Multi-Agent Engine)...")
        self.vector_store = vector_store_service
        self.llm = ChatOllama(model="llama3", temperature=0.1).bind(format="json")
        self.max_retries = 2 
        self.graph = self._build_graph()

    def _unwrap_hallucination(self, raw_dict: dict) -> dict:
        """Helper to catch local LLMs nesting output inside a 'properties' key."""
        if "properties" in raw_dict and len(raw_dict.keys()) == 1:
            return raw_dict["properties"]
        return raw_dict

    def semantic_evaluator(self, state: LeadState) -> dict:
        logger.info("Executing: Semantic Evaluator Agent")
        
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a KeaBuilder CRM Gatekeeper. Determine if the input is a valid sales lead or garbage/spam.
            Output EXACTLY this JSON format:
            {{
                "is_valid_sales_lead": true,
                "reason": "Brief explanation"
            }}"""),
            ("human", "Input: {lead_text}")
        ])
        
        chain = prompt | self.llm | parser
        raw_dict = chain.invoke({"lead_text": state["lead_text"]})
        raw_dict = self._unwrap_hallucination(raw_dict)
        
        decision = EvaluatorDecision(**raw_dict)
        
        logger.info(f"Evaluator Decision: Valid={decision.is_valid_sales_lead} | Reason: {decision.reason}")
        return {
            "is_valid_lead": decision.is_valid_sales_lead,
            "rejection_reason": decision.reason,
            "retry_count": 0,
            "feedback": ""
        }

    def memory_retriever(self, state: LeadState) -> dict:
        logger.info("Executing: Memory Retriever Agent")
        similar_leads = self.vector_store.get_best_match(state["lead_text"], top_k=3)
        context_string = "\n".join([f"- {lead['document']}" for lead in similar_leads])
        logger.info(f"Retrieved {len(similar_leads)} historical leads for context.")
        return {"rag_context": context_string}

    def worker_agent(self, state: LeadState) -> dict:
        logger.info(f"Executing: Worker Agent (Attempt {state.get('retry_count', 0) + 1})")
        
        parser = JsonOutputParser()
        feedback_instruction = ""
        if state.get("retry_count", 0) > 0:
            feedback_instruction = f"\nCRITICAL FEEDBACK FROM VALIDATOR. FIX THIS: {state['feedback']}"
            logger.warning(f"Worker applying feedback: {state['feedback']}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are KeaBuilder's CRM routing agent. Classify the lead intent (HOT/WARM/COLD) and draft a brief, personalized email.
            Historical Context:
            {context}
            {feedback}
            
            Output EXACTLY this JSON format:
            {{
                "classification": "HOT",
                "extracted_entity": "Gym",
                "drafted_email": "Your email here"
            }}"""),
            ("human", "Lead Text: {lead_text}")
        ])
        
        chain = prompt | self.llm | parser
        raw_dict = chain.invoke({
            "context": state.get("rag_context", ""),
            "feedback": feedback_instruction,
            "lead_text": state["lead_text"]
        })
        
        raw_dict = self._unwrap_hallucination(raw_dict)
        draft = DraftOutput(**raw_dict)
        
        logger.info(f"Worker Classification: {draft.classification}")
        return {"classification": draft.classification, "drafted_email": draft.drafted_email}

    def validator_agent(self, state: LeadState) -> dict:
        logger.info("Executing: Validator Agent (Quality Assurance)")
        
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Review this drafted email against the original user lead.
            Rules:
            1. Did it directly address the user's specific goal?
            2. Is the tone professional?
            3. Did it hallucinate pricing?
            If it passes, approve it. If it fails, provide explicit rewrite instructions.
            
            Output EXACTLY this JSON format:
            {{
                "is_approved": true,
                "feedback": "Feedback here or empty string if approved"
            }}"""),
            ("human", "Original Lead: {lead_text}\nDrafted Email: {drafted_email}")
        ])
        
        chain = prompt | self.llm | parser
        raw_dict = chain.invoke({
            "lead_text": state["lead_text"],
            "drafted_email": state["drafted_email"]
        })
        
        raw_dict = self._unwrap_hallucination(raw_dict)
        review = ValidatorDecision(**raw_dict)
        
        current_retries = state.get("retry_count", 0)
        if review.is_approved:
            logger.info("Validator APPROVED the draft.")
        else:
            logger.error(f"Validator REJECTED the draft. Feedback: {review.feedback}")
            
        return {
            "feedback": review.feedback,
            "retry_count": current_retries + 1 if not review.is_approved else current_retries
        }

# -------------------------------------------------------------------------
# 4. Graph Routing & Compilation
# -------------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(LeadState)
        
        workflow.add_node("evaluator", self.semantic_evaluator)
        workflow.add_node("retriever", self.memory_retriever)
        workflow.add_node("worker", self.worker_agent)
        workflow.add_node("validator", self.validator_agent)

        workflow.set_entry_point("evaluator")
        
        workflow.add_conditional_edges(
            "evaluator",
            lambda state: "continue" if state["is_valid_lead"] else "end",
            {"continue": "retriever", "end": END}
        )
        
        workflow.add_edge("retriever", "worker")
        workflow.add_edge("worker", "validator")

        def route_validation(state):
            if state["feedback"] == "":
                return "end" 
            if state["retry_count"] >= self.max_retries:
                logger.warning("Max retries reached. Forcing loop exit.")
                return "end"
            return "retry" 

        workflow.add_conditional_edges("validator", route_validation, {"end": END, "retry": "worker"})
        return workflow.compile()

    async def process_lead(self, text: str) -> dict:
        initial_state = {"lead_text": text}
        logger.info("========== NEW LEAD INGESTED INTO PIPELINE ==========")
        
        final_state = {}
        for output in self.graph.stream(initial_state):
            for node_name, state_update in output.items():
                logger.info(f"--- Completed Graph Node: [{node_name.upper()}] ---")
                final_state.update(state_update)
        
        logger.info("========== PIPELINE EXECUTION FINISHED ==========")
        
        if not final_state.get("is_valid_lead", False):
            raise ValueError(f"Rejected by Evaluator: {final_state.get('rejection_reason')}")
            
        return {
            "classification": final_state.get("classification"),
            "email": final_state.get("drafted_email"),
            "retries_required": final_state.get("retry_count")
        }