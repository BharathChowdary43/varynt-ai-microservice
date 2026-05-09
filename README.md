# 🧠 VARYNT AI Microservice: Agentic Lead Intelligence

An enterprise-grade, event-driven AI microservice built for the **VARYNT / KeaBuilder** platform. Engineered with **LangGraph**, **FastAPI**, and **ChromaDB**, this system handles intelligent lead qualification, self-correcting response generation, and semantic vector search—all completely decoupled from the main Node.js web server.

---

## 🏗️ The Philosophy: Modular Realism

This system is built on the principle of **Modular Realism**. It is designed to deliver high-performance, self-correcting AI behavior that respects immediate infrastructure constraints (zero API costs and strict data privacy via local models). At the same time, the business logic is entirely decoupled, making it ready to scale to managed cloud clusters the moment active usage and budget dictate.

---

## 🚀 Core Architecture & Workflows

### 1. The Multi-Agent Reflexion Loop (Lead Qualification)
Unlike standard, brittle linear LLM chains, this pipeline utilizes an **Actor-Critic StateGraph** to ensure production-level reliability and hallucination control.
<img width="5400" height="3002" alt="VARYNT_HLD" src="https://github.com/user-attachments/assets/661f184b-85eb-4370-b5ae-c572dccca10e" />

* **🛡️ Semantic Evaluator (Gatekeeper):** Analyzes the semantic density of incoming requests. It instantly filters out spam or low-effort inputs (e.g., "hi", "just looking"), preventing wasted compute cycles.
* **🧠 Memory Retriever (RAG):** Queries the vector database for historical "twins" of the current lead to provide contextual grounding.
* **⚙️ Worker Agent (Drafter):** Classifies the lead's intent (`HOT`, `WARM`, `COLD`) and drafts a hyper-personalized email response based on extracted entities.
* **🔎 Validator Agent (Supervisor):** The core of the Reflexion Loop. It strictly reviews the drafted email against the original lead. If it detects hallucinations (e.g., invented pricing) or a generic tone, it rejects the draft and loops it back to the Worker for a rewrite.

### 2. Semantic Memory Engine (Vector Retrieval)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (chosen for sub-millisecond, CPU-friendly inference).
* **Vector Store:** Local `ChromaDB` utilizing **HNSW (Hierarchical Navigable Small World)** indexing for highly optimized semantic matching of templates and historical leads.

---

## 📈 Scaling Roadmap

This MVP runs locally to maximize runway, but it is architected for immediate cloud deployment.

| Constraint Trigger | Current Local MVP | Production Scaling Target |
| :--- | :--- | :--- |
| **Active Usage / Concurrency Spikes** | `asyncio` background tasks | Introduce **Redis / RabbitMQ** to queue payloads, scale FastAPI Docker containers horizontally. |
| **High Read/Write Database IOPS** | Local SQLite-backed ChromaDB | Migrate to distributed **Pinecone** or **Milvus** cluster for metadata sharding. |
| **Complex Reasoning Needs / Budget** | Local `Ollama` (`Llama-3-8B`) | Swap endpoint to managed **OpenAI / Anthropic** APIs or dedicated `vLLM` GPU clusters. |

---

## 🛠️ Tech Stack
* **Language:** Python 3.11 (Strictly pinned for C++ binary compatibility with vector DBs)
* **Framework:** FastAPI
* **AI/Orchestration:** LangChain, LangGraph
* **Machine Learning:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Database:** ChromaDB
* **Local Inference:** Ollama (JSON Mode)

---

## 💻 Getting Started (Local Development)

### 1. Prerequisites
* Install [Python 3.11](https://www.python.org/downloads/release/python-3110/)
* Install [Ollama](https://ollama.com/)
* Install [uv](https://github.com/astral-sh/uv) (Fast Python Package Installer)

### 2. Boot Local LLM
Open a terminal and start the Ollama server with the Llama 3 model:
```bash
ollama run llama3
```
*(Keep this terminal running in the background).*

### 3. Environment Setup
Clone the repository and set up your virtual environment:
```bash
git clone [https://github.com/BharathChowdary43/varynt-ai-microservice.git](https://github.com/BharathChowdary43/varynt-ai-microservice.git)
cd varynt-ai-microservice

# Create and activate environment
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Run the Microservice
Start the FastAPI gateway:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000/docs` to interact with the Swagger UI and test the endpoints.

---

## 🐳 Docker Deployment (Microservice Approach)

To run the application in an isolated container while utilizing your host machine's Ollama runner for GPU/CPU optimization:

```bash
# Build the image
docker build -t varynt-ai-service .

# Run the container (Maps host.docker.internal to access local Ollama)
docker run -p 8000:8000 varynt-ai-service
```

---

## 📜 Logging & Observability
All agent thoughts, state transitions, and validation critiques are logged locally to `varynt_agent_trace.log`. In a production environment, this is designed to be easily piped into **LangSmith** and **Datadog**.

---
*Architected and developed by [Sri Bharath Pentela](https://github.com/BharathChowdary43) for VARYNT.*
