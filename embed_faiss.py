import os
from utils.index_w_faiss import FaissClient

def process_pdfs_and_embed(pdf_directory: str, index_path: str):
    # Initialize FaissClient
    faiss_client = FaissClient(index_path)

    # Process each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing {filename}...")

            # Process the file using FaissClient
            faiss_client.process_file(pdf_path)

    print("Finished processing all PDFs and embedding into FAISS index.")

if __name__ == "__main__":
    # Set the directory containing your PDF files
    pdf_directory = "vumc_pdfs"
    
    # Set the path for your FAISS index
    index_path = "surgical_faiss_index"

    # Run the process
    process_pdfs_and_embed(pdf_directory, index_path)