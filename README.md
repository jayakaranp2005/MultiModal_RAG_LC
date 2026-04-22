# MultiModal_RAG_LC

A simple starter repository for experimenting with **Multi-Modal Retrieval-Augmented Generation (RAG)** using **LangChain**.

## What this repo is for
- Build a RAG pipeline that can retrieve and ground answers from documents.
- Experiment with multi-modal inputs (e.g., text + images) and embeddings.
- Prototype quickly with LangChain components and common vector stores.

## Quick start
1. Clone the repo:
   ```bash
   git clone https://github.com/jayakaranp2005/MultiModal_RAG_LC.git
   cd MultiModal_RAG_LC
   ```
2. Create and activate a virtual environment (recommended).
3. Install dependencies (adjust based on your project files):
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script / notebook (see the repo files).

## Configuration
Most RAG projects require API keys (e.g., OpenAI) and settings for the vector store.

Create a `.env` file (do **not** commit secrets):
```env
OPENAI_API_KEY=your_key_here
```

## Project structure
This will vary as the project evolves. Common folders you may see:
- `data/` – source documents
- `src/` – application code
- `notebooks/` – experiments

## Notes
- Keep large datasets and generated vector indexes out of git.
- Add a license if you plan to share or reuse the code.

## License
No license specified yet.