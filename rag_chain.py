from langchain.chains import RetrievalQA # type: ignore
from langchain.llms import OpenAI # type: ignore
from langchain_community.llms import Gemini  # type: ignore #
from dotenv import load_dotenv # type: ignore
import os

load_dotenv()


class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        # Use OpenAI if key exists
        if os.getenv("OPENAI_API_KEY"):
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        # Use Gemini if key exists
        elif os.getenv("GEMINI_API_KEY"):
            return Gemini(api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def create_chain(self):
        # Create a retriever from FAISS vectorstore
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Build the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain


if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    # Load and split documents
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    # Create vector store (FAISS)
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    # Build RAG chain
    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    # Ask a question
    query = "What is the capital of France?"
    result = qa_chain({"query": query})

    print(f"Answer: {result['result']}")
    print("\nSource Documents:")
    for doc in result['source_documents']:
        print("-", doc.page_content[:200], "...")