# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents using local or cloud-based LLMs.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) combines two steps:
1. **Retrieval** — finds relevant chunks from your document knowledge base using semantic search
2. **Generation** — feeds those chunks to an LLM to generate an accurate, context-aware answer

This means your chatbot can answer questions about your own data, even if the LLM was never trained on it.

---

## Project Structure

```
RAG_Chatbot/
├── chatbot.py              # Main entry point — RAG chain + interactive chatbot
├── document_processor.py   # Loads and splits documents into chunks
├── embedding_indexer.py    # Creates and saves FAISS vector store
├── rag_chain.py            # Defines the RAG pipeline
├── data/
│   └── sample_text.txt     # Your knowledge base document
├── vector_db/              # Auto-generated FAISS index (created on first run)
├── .env                    # API keys (not committed to git)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Quickstart

### 1. Prerequisites

```bash
brew install python@3.11
```

### 2. Clone and set up environment

```bash
git clone <your-repo-url>
cd RAG_Chatbot

python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your document

Place your knowledge base text file at:
```
data/sample_text.txt
```

### 4. Configure your LLM

Create a `.env` file in the project root and add your preferred API key:

```properties
# Option A: Ollama (free, local — recommended)
# No API key needed, see Ollama setup below

# Option B: Gemini (free tier available)
GEMINI_API_KEY=your_gemini_api_key_here

# Option C: OpenAI
# OPENAI_API_KEY=your_openai_api_key_here

# Option D: Fireworks AI
# FIREWORKS_API_KEY=your_fireworks_api_key_here
```

### 5. Run the chatbot

```bash
python chatbot.py
```

---

## LLM Options

### 🦙 Ollama (Recommended — Free & Local)

Runs entirely on your machine. No API key, no quotas, no internet required after setup.

```bash
# Install Ollama
brew install ollama

# Start Ollama
ollama serve &

# Pull a model
ollama pull llama3.2

# In chatbot.py, get_llm() is already configured for Ollama
```

> **Note:** `llama3.2` requires ~8GB RAM. For lower-spec machines use `llama3.2:1b`.

---

### ☁️ Cloud LLM Options

| Provider | Free Tier | Get API Key |
|---|---|---|
| Google Gemini | 1,500 req/day | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Groq | 14,400 req/day | [console.groq.com](https://console.groq.com) |
| Fireworks AI | Limited free tier | [fireworks.ai](https://fireworks.ai/account/api-keys) |
| OpenAI | Paid | [platform.openai.com](https://platform.openai.com/api-keys) |

---

## How It Works

```
User Query
    │
    ▼
HuggingFace Embeddings          ← sentence-transformers/all-MiniLM-L6-v2
    │
    ▼
FAISS Vector Search             ← retrieves top-3 relevant chunks
    │
    ▼
RetrievalQA Chain (LangChain)   ← combines chunks + query into a prompt
    │
    ▼
LLM (Ollama / Gemini / OpenAI)  ← generates the final answer
    │
    ▼
Chatbot Response
```

---

## Dependencies

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

Key packages used:

| Package | Purpose |
|---|---|
| `langchain` | RAG chain orchestration |
| `langchain-community` | FAISS, HuggingFace, Fireworks integrations |
| `langchain-ollama` | Ollama LLM integration |
| `langchain-openai` | OpenAI LLM integration |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local embedding model |
| `google-genai` | Gemini API client |
| `python-dotenv` | Load `.env` API keys |

To regenerate `requirements.txt`:
```bash
pipreqs .
```

---

## Rebuilding the Vector Store

If you update your document, delete the cached index and rerun:

```bash
rm -rf vector_db/
python chatbot.py
```

---

## Security

⚠️ **Never commit your `.env` file.** Add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

---

## Example Usage

```
Chatbot is ready! Type 'exit', 'quit', or 'bye' to stop.
You: What is Aurion?
Chatbot: Aurion is ...
You: Who are the members?
Chatbot: The members are ...
You: bye
Chatbot: Goodbye!
```