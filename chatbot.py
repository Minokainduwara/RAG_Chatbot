import os
from dotenv import load_dotenv # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.llms import OpenAI # type: ignore
from langchain_community.llms import Gemini  # type: ignore # For Google Gemini

from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from langchain_community.vectorstores import FAISS # type: ignore

load_dotenv()


class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif os.getenv("GEMINI_API_KEY"):
            return Gemini(api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain


class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain({"query": user_input})
            answer = response['result']
            return answer
        except Exception as e:
            return f"An error occurred: {str(e)}"


def main():
    # Load FAISS vectorstore if exists, else rebuild
    if os.path.exists("vector_db/index.faiss"):
        print("Loading existing FAISS vectorstore...")
        vectorstore = FAISS.load_local("vector_db")
    else:
        print("Building FAISS vectorstore from documents...")
        processor = DocumentProcessor("data/sample_text.txt")
        texts = processor.load_and_split()

        indexer = EmbeddingIndexer()
        vectorstore = indexer.create_vectorstore(texts)

    # Build RAG chain
    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    # Initialize chatbot
    chatbot = Chatbot(qa_chain)

    # Interactive loop
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