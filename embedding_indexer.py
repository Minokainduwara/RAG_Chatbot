# embedding_indexer.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_vectorstore(self, texts):
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        vectorstore.save_local("vector_db")
        return vectorstore

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print(f"Vector store created successfully with {len(texts)} chunks.")