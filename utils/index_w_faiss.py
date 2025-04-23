from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict, Tuple

class FaissClient:

    def __init__(self, index_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_file(self, file_path: str) -> str:
        md_engine = MarkItDown(enable_plugins=False)
        try:
            result = md_engine.convert(file_path)
            return result.text_content
        except Exception as e:
            print(f"Error occurred while extracting text from PDF: {e}")
            return None
        
    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks

    def get_vectors(self, chunks: List[str]) -> np.ndarray:
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

    def save_index_info(self, chunks: List[str], ids: List[int], metadata: List[Dict]):
        index_json_file = os.path.basename(self.index_path) + ".json"
        if not os.path.exists(index_json_file):
            print("No matching index info found. Creating new one...")
            index_info = {
                "index_path": self.index_path,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "passages": chunks,
                "ids": ids,
                "metadata": metadata,
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
            new_ids = [max(index_info["ids"]) + i + 1 for i in range(len(ids))]
            index_info['ids'].extend(new_ids)
            index_info['metadata'].extend(metadata)
            with open(index_json_file, "w") as f:
                json.dump(index_info, f, indent=2)
            print("Updated index info saved to index_info.json")

    def process_file(self, file_path: str):
        # Extract text from file
        text = self.extract_text_from_file(file_path)
        if text is None:
            print(f"Failed to extract text from {file_path}")
            return

        # Chunk the text
        chunks = self.chunk_text(text)

        # Get vectors for chunks
        vectors = self.get_vectors(chunks)

        # Write vectors to FAISS index
        self.write_with_faiss(vectors)

        # Prepare metadata
        metadata = []
        start_line = 1
        for chunk in chunks:
            end_line = start_line + chunk.count('\n')
            metadata.append({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line
            })
            start_line = end_line + 1

        # Save index info
        ids = list(range(len(chunks)))
        self.save_index_info(chunks, ids, metadata)

        print(f"Successfully processed {file_path} and added to the index.")

    def process_directory(self, directory_path: str):
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pdf'):  # Assuming we're only processing PDF files
                    file_path = os.path.join(root, file)
                    self.process_file(file_path)


class FaissReader:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self.index_info = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_index()
        self.load_index_info()

    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    def load_index_info(self):
        index_json_file = f"{self.index_path}.json"
        with open(index_json_file, "r") as f:
            self.index_info = json.load(f)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, any]]:
        query_vector = self.model.encode([query_text], normalize_embeddings=True)
        scores, indices = self.index.search(query_vector.astype("float32"), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            chunk = self.index_info["passages"][idx]
            metadata = self.index_info["metadata"][idx]
            results.append({
                "chunk": chunk,
                "score": float(scores[0][i]),
                "file_path": metadata["file_path"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"]
            })
        
        return results
    
    def make_text_from_results(self, results: List[Dict[str, any]]) -> str:
        text = ""
        for result in results:
            text += f"\n---\nChunk:\n{result['chunk']}\n---\nFile Path: {result['file_path']}\nStart Line: {result['start_line']}\nEnd Line: {result['end_line']}\n\n"
        return text.strip()
    
    def search(self, query_text: str, k: int = 5) -> str:
        return self.make_text_from_results(self.query(query_text, k))

    def query_with_context(self, query_text: str, k: int = 5, context_size: int = 1) -> List[Dict[str, any]]:
        initial_results = self.query(query_text, k)
        
        contextualized_results = []
        for result in initial_results:
            context_before, context_after = self.get_context(result, context_size)
            contextualized_results.append({
                **result,
                "context_before": context_before,
                "context_after": context_after
            })
        
        return contextualized_results

    def get_context(self, result: Dict[str, any], context_size: int) -> Tuple[List[str], List[str]]:
        file_path = result["file_path"]
        start_line = result["start_line"]
        end_line = result["end_line"]
        
        context_before = []
        context_after = []
        
        for idx, metadata in enumerate(self.index_info["metadata"]):
            if metadata["file_path"] == file_path:
                if metadata["end_line"] < start_line and len(context_before) < context_size:
                    context_before.insert(0, self.index_info["passages"][idx])
                elif metadata["start_line"] > end_line and len(context_after) < context_size:
                    context_after.append(self.index_info["passages"][idx])
        
        return context_before, context_after