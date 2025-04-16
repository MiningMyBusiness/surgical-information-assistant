import os
from typing import List
from langchain_core.documents import Document
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import uuid


class DocumentExtractor:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.md_engine = MarkItDown(enable_plugins=False)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def run(self, file_path: str) -> List[Document]:
        text = self.extract_text_from_file(file_path)
        if text is None:
            return []
        
        documents = self.create_documents(file_path, text)
        return documents
    
    def extract_text_from_file(self, file_path: str) -> str:
        try:
            result = self.md_engine.convert(file_path)
            return result.text_content
        except Exception as e:
            print(f"Error occurred while extracting text from PDF: {e}")
            return None
        
    def create_documents(self, file_path: str, text: str) -> List[Document]:
        chunks = self.splitter.split_text(text)
        base_filename = os.path.basename(file_path)
        documents = []
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": base_filename})
            documents.append(doc)
        return documents
    

class MilvusClient:

    def __init__(self, collection_name: str, milvus_directory: str="./milvus.db"):
        self.collection_name = collection_name
        self.milvus_directory = milvus_directory
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def write_to_milvus(self, documents: List[Document], drop_old: bool = True):
        vectorstore = Milvus(
            embedding_function=self.embedding,
            collection_name=self.collection_name,
            connection_args={"uri": self.milvus_directory},
            drop_old=drop_old,
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )
        self.delete_lock_file()
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        vectorstore.add_documents(documents, ids=doc_ids)

    def read_from_milvus(self, query: str, k: int=4) -> List[Document]:
        vectorstore = Milvus(
            embedding_function=self.embedding,
            collection_name=self.collection_name,
            connection_args={"uri": self.milvus_directory},
            drop_old=False,
        )
        self.delete_lock_file()
        query_results = vectorstore.max_marginal_relevance_search(query=query, k=k)
        result_texts = [result.page_content + "\nSource:" + result.metadata['source'] for result in query_results]
        text = "\n\n".join(result_texts)
        return text
    
    def delete_lock_file(self):
        grandparent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lock_file_path = os.path.join(grandparent_directory, "milvus.db.lock")
        if os.path.exists(lock_file_path):
            print(f"Lock file {lock_file_path} exists. Deleting it.")
            os.remove(lock_file_path)

        