# document_processor.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        loader = TextLoader(self.file_path, encoding="utf-8")
        documents = loader.load()
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(documents)

if __name__ == "__main__":
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    print(f"Processed {len(texts)} text chunks")