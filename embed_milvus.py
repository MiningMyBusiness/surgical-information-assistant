import os
from utils.write_to_milvus import DocumentExtractor, MilvusClient

def process_pdfs_and_embed(pdf_directory: str, collection_name: str):
    # Initialize the DocumentExtractor and MilvusClient
    extractor = DocumentExtractor()
    milvus_client = MilvusClient(collection_name)

    # Process each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing {filename}...")

            # Extract text and create documents
            documents = extractor.run(pdf_path)

            if documents:
                # Write documents to Milvus
                milvus_client.write_to_milvus(documents, drop_old=False)
                print(f"Successfully embedded {len(documents)} chunks from {filename}")
            else:
                print(f"No content extracted from {filename}")

    print("Finished processing all PDFs and embedding into Milvus.")

if __name__ == "__main__":
    # Set the directory containing your PDF files
    pdf_directory = "vumc_pdfs"
    
    # Set the name for your Milvus collection
    collection_name = "surgical_information"

    # Run the process
    process_pdfs_and_embed(pdf_directory, collection_name)