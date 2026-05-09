# VARYNT AI-SaaS Microservice

This repository contains the backend AI orchestration layer developed for the Dream Reflection Media / KeaBuilder technical assessment. 

## 🏗️ Architecture Overview
To ensure high availability and prevent AI inference tasks from blocking KeaBuilder's core Node.js event loop, this system utilizes a fully decoupled, event-driven microservice architecture. 

**Core Tech Stack:**
* **Routing & API:** Python, FastAPI
* **Agentic Orchestration:** Custom Router, Evaluator, and Execution Agents
* **Vector Memory:** ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Queueing & Scaling:** Redis Message Queue, Docker (simulated for deployment)

## ✨ Key Workflows

### 1. Intelligent Lead Classification
A multi-agent pipeline designed to handle raw CRM inputs:
* **Evaluator Agent:** Acts as a gatekeeper. Evaluates semantic density to prevent expensive LLM calls on garbage data (e.g., "help me"). Triggers a fallback webhook if clarity is needed.
* **Execution Agent:** Injects valid inputs into a strict, few-shot prompt to classify intent (HOT, WARM, COLD) and draft personalized, context-aware responses in strict JSON.

### 2. Semantic Similarity Search
Replaces brittle keyword matching with dense vector retrieval:
* Generates localized embeddings using `all-MiniLM-L6-v2`.
* Queries a persistent ChromaDB instance to calculate Cosine Similarity.
* Returns the highest percentage match for funnel templates or previous inputs.

### 3. Asynchronous Multi-Modal Routing
Acts as a switchboard for media generation:
* Routes Text-to-Image requests (Replicate), Video (Runway), and Voice (ElevenLabs).
* Utilizes asynchronous webhook patterns (HTTP 202 Accepted) to prevent UI latency and freezing during long generation tasks.
* Supports LoRA `.safetensors` adapter injection for brand/face consistency.

<img width="5400" height="3002" alt="VARYNT_HLD" src="https://github.com/user-attachments/assets/318fc6e9-b3e4-4ba2-b735-08361b91c105" />

## 🚀 Setup & Installation

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/varynt-ai.git](https://github.com/yourusername/varynt-ai.git)

# 2. Navigate into the directory
cd varynt-ai

# 3. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# 4. Run the FastAPI server locally
uvicorn main:app --reload
