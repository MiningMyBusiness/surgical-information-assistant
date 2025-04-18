from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import uuid

class FaissClient:

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_file(self, file_path: str) -> str:
        md_engine = MarkItDown(enable_plugins=False)
        try:
            result = md_engine.convert(file_path)
            return result.text_content
        except Exception as e:
            print(f"Error occurred while extracting text from PDF: {e}")
            return None
        
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks

    def get_vectors(self, chunks: list[str]) -> np.ndarray:
        vectors = self.model.encode(chunks, normalize_embeddings=True)
        return np.array(vectors)
    
    def write_with_faiss(self, vectors: np.ndarray):
        if os.path.exists(self.index_path):
            print("Reading existing index...")
            index = faiss.read_index(self.index_path)
        else:
            print("Creating new index...")
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype("float32"))
        faiss.write_index(index, self.index_path)

    def save_index_info(self, chunks: list[str], ids: list[int]):
        index_json_file = os.path.basename(self.index_path) + ".json"
        if not os.path.exists(index_json_file):
            print("No matching index info found. Creating new one...")
            index_info = {
                "index_path": self.index_path,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "passages": chunks,
                "ids": ids,
                "model_name": "all-MiniLM-L6-v2",
            }
            with open(index_json_file, "w") as f:
                json.dump(index_info, f, indent=2)
            print("Index info saved to index_info.json")
        else:
            print("Index info already exists. Loading index and appending new passages along with modified ids...")
            with open(index_json_file, "r") as f:
                index_info = json.load(f)
            index_info["passages"].extend(chunks)
            ids = [this_id + max(index_info["ids"]) + 1 for this_id in ids]
            index_info['ids'].extend(ids)