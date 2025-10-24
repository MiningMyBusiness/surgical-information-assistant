from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict, Tuple
import re
from collections import Counter
import string

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

    def process_dataset(self, dataset, content_keys: List[str], metadata_keys: List[str]):
        all_chunks = []
        all_metadata = []

        for item in dataset:
            # 1. Combine content from specified keys
            content_list = [str(item[key]) for key in content_keys if key in item and item[key] is not None]
            full_content = "\n".join(content_list)

            if not full_content:
                continue

            # 2. Chunk the content
            chunks = self.chunk_text(full_content)
            all_chunks.extend(chunks)

            # 3. Create metadata for each chunk
            base_metadata = {key: item.get(key) for key in metadata_keys}
            for chunk in chunks:
                chunk_metadata = base_metadata.copy()
                all_metadata.append(chunk_metadata)
        
        if not all_chunks:
            print("No content to process from the dataset.")
            return

        # 4. Get vectors for all chunks in a batch
        vectors = self.get_vectors(all_chunks)

        # 5. Write vectors to FAISS index
        self.write_with_faiss(vectors)

        # 6. Save index info
        ids = list(range(len(all_chunks)))
        self.save_index_info(all_chunks, ids, all_metadata)

        print(f"Successfully processed {len(dataset)} items from the dataset and added to the index.")


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

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return re.findall(r'\b\w+\b', text)

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def _mmr_rerank(self, query_text: str, results: List[Dict[str, any]], k: int, lambda_mult: float = 0.7) -> List[Dict[str, any]]:
        if not results:
            return []

        query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0]
        
        # Get embeddings for all result chunks
        result_chunks = [res["chunk"] for res in results]
        result_embeddings = self.model.encode(result_chunks, normalize_embeddings=True)

        # Initialize MMR
        unranked_results = list(zip(results, result_embeddings))
        ranked_results = []

        # Select the first result (most relevant to the query)
        first_result_idx = np.argmax([cosine_similarity(query_embedding, emb) for emb in result_embeddings])
        ranked_results.append(unranked_results.pop(first_result_idx)[0])

        # Iteratively select the rest
        while unranked_results and len(ranked_results) < k:
            mmr_scores = []
            for res, res_emb in unranked_results:
                relevance_score = cosine_similarity(query_embedding, res_emb)
                
                # Calculate similarity to already selected results
                max_similarity = 0
                if ranked_results:
                    ranked_embeddings = self.model.encode([r["chunk"] for r in ranked_results], normalize_embeddings=True)
                    max_similarity = np.max([cosine_similarity(res_emb, ranked_emb) for ranked_emb in ranked_embeddings])
                
                mmr_score = lambda_mult * relevance_score - (1 - lambda_mult) * max_similarity
                mmr_scores.append(mmr_score)

            best_idx = np.argmax(mmr_scores)
            ranked_results.append(unranked_results.pop(best_idx)[0])

        return ranked_results

    def _fast_rerank(self, query_text: str, results: List[Dict[str, any]], k: int) -> List[Dict[str, any]]:
        if not results:
            return []

        query_tokens = self._tokenize(query_text)
        query_unigrams = self._get_ngrams(query_tokens, 1)
        query_bigrams = self._get_ngrams(query_tokens, 2)

        reranked_results = []
        for res in results:
            chunk_text = res["chunk"]
            chunk_tokens = self._tokenize(chunk_text)
            
            # Keyword match score
            keyword_score = sum(1 for token in query_tokens if token in chunk_tokens)
            
            # Unigram overlap
            chunk_unigrams = self._get_ngrams(chunk_tokens, 1)
            unigram_overlap = len(set(query_unigrams) & set(chunk_unigrams))
            
            # Bigram overlap
            chunk_bigrams = self._get_ngrams(chunk_tokens, 2)
            bigram_overlap = len(set(query_bigrams) & set(chunk_bigrams))
            
            # Combine scores
            combined_score = (0.4 * keyword_score) + (0.2 * unigram_overlap) + (0.4 * bigram_overlap)
            
            res['rerank_score'] = combined_score
            reranked_results.append(res)

        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        for res in reranked_results:
            del res['rerank_score']

        return reranked_results[:k]

    def query(self, query_text: str, k: int = 5, rerank: str = 'fast') -> List[Dict[str, any]]:
        # Fetch more results initially to have a good pool for MMR
        initial_k = k * 4
        query_vector = self.model.encode([query_text], normalize_embeddings=True)
        scores, indices = self.index.search(query_vector.astype("float32"), initial_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.index_info["passages"]):
                continue  # Skip invalid indices
            chunk = self.index_info["passages"][idx]
            metadata = self.index_info["metadata"][idx]
            result_item = {
                "chunk": chunk,
                "score": float(scores[0][i]),
            }
            result_item.update(metadata)  # Add all metadata to the result
            results.append(result_item)
        
        if rerank == 'mmr':
            results = self._mmr_rerank(query_text, results, k=k)
        elif rerank == 'fast':
            results = self._fast_rerank(query_text, results, k=k)
            
        return results[:k]
    
    def make_text_from_results(self, results: List[Dict[str, any]]) -> str:
        text = ""
        for result in results:
            text += f"\n---\nChunk:\n{result['chunk']}\n"
            text += f"Score: {result['score']:.4f}\n"
            # Dynamically add other metadata
            for key, value in result.items():
                if key not in ['chunk', 'score']:
                    text += f"{key.replace('_', ' ').title()}: {value}\n"
            text += "\n"
        return text.strip()
    
    def search(self, query_text: str, k: int = 5, rerank: str = 'fast') -> str:
        results = self.query(query_text, k, rerank=rerank)
        return self.make_text_from_results(results)

    def query_with_context(self, query_text: str, k: int = 5, rerank: str = 'fast', context_size: int = 1) -> List[Dict[str, any]]:
        initial_results = self.query(query_text, k, rerank=rerank)
        
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
        if 'file_path' not in result or 'start_line' not in result or 'end_line' not in result:
            return [], []

        file_path = result["file_path"]
        start_line = result["start_line"]
        end_line = result["end_line"]
        
        context_before = []
        context_after = []
        
        # This assumes that passages from the same file are somewhat contiguous in the index
        # which is true for process_file and process_directory, but not guaranteed for datasets.
        for idx, metadata in enumerate(self.index_info["metadata"]):
            if metadata.get("file_path") == file_path:
                if metadata.get("end_line", -1) < start_line and len(context_before) < context_size:
                    context_before.insert(0, self.index_info["passages"][idx])
                elif metadata.get("start_line", -1) > end_line and len(context_after) < context_size:
                    context_after.append(self.index_info["passages"][idx])
        
        return context_before, context_after