import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from typing import Optional, List
import google.genai as genai

load_dotenv()

# -----------------------------
# Custom Gemini wrapper
# -----------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM):
    """LangChain-compatible wrapper for Google Gemini using google-genai."""
    model: str = "gemini-1.5-flash"  # FIX 1: was "gemini-1.5" (invalid model name)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # FIX 2: correct API is generate_content(), not client.chat.create()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text  # FIX 3: was response.last (doesn't exist)

# -----------------------------
# RAG Chain
# -----------------------------
class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif os.getenv("GEMINI_API_KEY"):
            return GeminiLLM()
        elif os.getenv("FIREWORKS_API_KEY"):
            from langchain_community.llms import Fireworks
            return Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found in .env!")

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

# -----------------------------
# Chatbot
# -----------------------------
class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain({"query": user_input})
            return response['result']
        except Exception as e:
            return f"An error occurred: {str(e)}"

# -----------------------------
# Main
# -----------------------------
def main():
    if os.path.exists("vector_db/index.faiss"):
        print("Loading existing FAISS vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # FIX 4: allow_dangerous_deserialization=True is required in newer LangChain
        vectorstore = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building FAISS vectorstore from documents...")
        processor = DocumentProcessor("data/sample_text.txt")
        texts = processor.load_and_split()
        indexer = EmbeddingIndexer()
        vectorstore = indexer.create_vectorstore(texts)

    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()
    chatbot = Chatbot(qa_chain)

    print("Chatbot is ready! Type 'exit', 'quit', or 'bye' to stop.")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response}")
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye!")

if __name__ == "__main__":
    main()