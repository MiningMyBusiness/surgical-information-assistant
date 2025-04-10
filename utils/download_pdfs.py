import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time

# Base URL of the Table of Contents
base_url = "https://www.vumc.org/global-surgical-atlas/"

# Folder to save PDFs
download_folder = "vumc_pdfs"
os.makedirs(download_folder, exist_ok=True)

# Helper function to download PDF
def download_pdf(pdf_url):
    pdf_name = os.path.basename(urlparse(pdf_url).path)
    pdf_path = os.path.join(download_folder, pdf_name)
    
    if os.path.exists(pdf_path):
        print(f"Already downloaded: {pdf_name}")
        time.sleep(1)  # Avoid overwhelming the server with requests
        return

    print(f"Downloading {pdf_url}...")
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {pdf_path}")
    else:
        print(f"Failed to download: {pdf_url}")
    time.sleep(3)  # Avoid overwhelming the server with requests

def download_vumc_pdfs():
    # Step 1: Get all links from the main page
    toc_response = requests.get(base_url)
    toc_soup = BeautifulSoup(toc_response.content, "html.parser")

    # Step 2: Visit each link and search for PDFs
    for a_tag in toc_soup.find_all("a", href=True):
        link_url = urljoin(base_url, a_tag["href"])
        
        if link_url.endswith(".pdf"):
            # Direct link to PDF
            download_pdf(link_url)
        else:
            # Possibly an intermediate HTML page
            try:
                subpage_response = requests.get(link_url)
                subpage_soup = BeautifulSoup(subpage_response.content, "html.parser")
                for sub_a_tag in subpage_soup.find_all("a", href=True):
                    sub_href = sub_a_tag["href"]
                    if sub_href.endswith(".pdf"):
                        pdf_url = urljoin(link_url, sub_href)
                        download_pdf(pdf_url)
            except Exception as e:
                print(f"Error processing {link_url}: {e}")
