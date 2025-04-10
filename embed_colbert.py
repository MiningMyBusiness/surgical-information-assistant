import os
import fitz  # PyMuPDF
from ragatouille import RAGPretrainedModel
from pathlib import Path
import uuid

# Directories
text_folder = "pdf_texts"
index_name = "surgical_colbert_index"

# Step 1: Extract text from PDFs and create passages
def extract_passages_from_text():
    filenames = []
    full_texts = []
    for filename in os.listdir(text_folder):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
            full_text = f.read()

        filenames.append(filename)
        full_texts.append(full_text)
    return filenames, full_texts

# Step 2: Load ColBERT (RAGatouille) and index the passages
def embed_with_colbert(filenames, full_texts):
    # Load pre-trained ColBERT model (MContriever by default)
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    # Index passages
    index_path = RAG.index(
        index_name=index_name,
        collection=full_texts,
        document_ids=filenames,
    )
    return index_path


if __name__ == "__main__":
    # Run the full process
    filenames, passages = extract_passages_from_text()
    embed_with_colbert(filenames, passages, index_name)
