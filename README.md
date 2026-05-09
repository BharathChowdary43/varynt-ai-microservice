# VARYNT AI Microservice: Multi-Agent Lead Engine

An enterprise-grade AI pipeline for lead qualification and semantic search using **LangGraph** and **ChromaDB**.

## 🏗️ Architecture: The Reflexion Loop
Unlike standard linear chains, this system utilizes a **StateGraph-based Reflexion Loop (Actor-Critic pattern)**:
1. **Semantic Evaluator:** Acts as a gatekeeper to filter low-quality/spam inputs.
2. **Memory Retriever:** Performs RAG via ChromaDB to ground responses in historical context.
3. **Worker Agent:** Drafts personalized responses and classifies leads (HOT/WARM/COLD).
4. **Validator Agent:** Reviews the draft. If it fails quality checks, it triggers a rewrite loop (capped at 2 retries).

## 🚀 Scaling Roadmap (Future Improvements)
- **Infrastructure:** Currently optimized for local-first inference (Ollama/Llama3) to prioritize privacy and zero API costs.
- **Production Move:** Ready for migration to **LangSmith** for observability and **Pinecone** for high-scale vector retrieval.
- **Async Handling:** Designed for future decoupling via **RabbitMQ** or **Redis** for high-concurrency environments.

## 🛠️ Setup
1. `uv venv --python 3.11`
2. `uv pip install -r requirements.txt`
3. Ensure **Ollama** is running `llama3` locally.
4. `uvicorn app.main:app --reload`