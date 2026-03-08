from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore


class EmbeddingIndexer:
    def __init__(self):
        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_vectorstore(self, texts):
        # Create FAISS vector database from documents
        vectorstore = FAISS.from_documents(texts, self.embeddings)

        # Save the vector database locally
        vectorstore.save_local("vector_db")

        return vectorstore


if __name__ == "__main__":
    from document_processor import DocumentProcessor

    # Load and split documents
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    # Create embeddings and FAISS index
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    print(f"Vector store created successfully with {len(texts)} chunks.")