import os
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
from utils.llms import init_llm
import asyncio
from typing import List, Dict
import time
import logging
import json
from asyncio import TimeoutError
import aiohttp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download("punkt")
nltk.download('punkt_tab')

# Rate limiting constants
MAX_REQUESTS_PER_MINUTE = 250
SEMAPHORE_VALUE = 50  # Allow up to 50 concurrent requests
TIMEOUT_SECONDS = 60  # Timeout for each LLM request

# Folder with PDFs
pdf_folder = "vumc_pdfs"
text_folder = "pdf_texts"
os.makedirs(text_folder, exist_ok=True)

start_time = [time.time()]


async def async_generate_qa_pair(chunk: str) -> List[Dict[str, str]]:
    llm = init_llm('qwen2.5-7b')
    prompt = get_prompt(chunk)
    logging.debug(f"Generating QA pair for chunk: {chunk[:50]}...")
    try:
        async with aiohttp.ClientSession() as session:
            response = await llm.ainvoke(prompt, session=session)
        return parse_response(response.content)
    except TimeoutError:
        logging.error(f"Timeout occurred while generating QA pair for chunk: {chunk[:50]}...")
        return []
    except Exception as e:
        logging.error(f"Error occurred while generating QA pair: {str(e)}")
        return []


# Modify the generate_dataset function to use asyncio
async def async_generate_dataset():
    dataset = []
    semaphore = asyncio.Semaphore(SEMAPHORE_VALUE)
    request_count = 0

    async def process_file(filename):
        nonlocal request_count
        if filename.endswith(".txt"):
            logging.info(f"Processing file: {filename}")
            async with semaphore:
                try:
                    with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
                        text = f.read()
                    qa_pairs = await generate_qa_pairs_from_text(text, semaphore)
                    request_count += len(qa_pairs)
                    
                    # Check if we need to pause to respect rate limit
                    elapsed_time = time.time() - start_time[0]
                    if elapsed_time < 60 and request_count >= MAX_REQUESTS_PER_MINUTE:
                        sleep_time = 60 - elapsed_time
                        logging.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                        await asyncio.sleep(sleep_time)
                        start_time[0] = time.time()
                        request_count = 0
                    
                    logging.info(f"Generated {len(qa_pairs)} QA pairs for {filename}")
                    return qa_pairs
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {str(e)}")
                    return []

    tasks = [process_file(filename) for filename in os.listdir(text_folder)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        if result:
            dataset.extend(result)
    
    logging.info(f"Total QA pairs generated: {len(dataset)}")
    return dataset


# 1. Extract text from each PDF
def extract_text_from_pdfs():
    logging.info("Starting text extraction from PDFs")
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            logging.info(f"Extracting text from {filename}")
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            # Save extracted text
            base_name = Path(filename).stem
            with open(os.path.join(text_folder, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            logging.info(f"Extracted text saved to {base_name}.txt")
    logging.info("Finished extracting text from all PDFs")

# 2. Generate QA pairs from a text chunk
async def generate_qa_pairs_from_text(text: str, semaphore: asyncio.Semaphore, chunk_size: int = 5) -> List[Dict[str, str]]:
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]
    logging.info(f"Generated {len(chunks)} chunks from text")

    async def process_chunk(chunk):
        async with semaphore:
            return await async_generate_qa_pair(chunk)

    tasks = [process_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    qa_pairs = [pair for result in results for pair in result]
    logging.info(f"Generated {len(qa_pairs)} QA pairs from text")
    return qa_pairs


# 2.5. Get prompt
def get_prompt(text_chunk: str) -> str:
    prompt = f"""You are a medical reasoning engine. Given a medical or medically-related text passage, you must generate question-answer pairs. The questions MUST be understandable on its own without needing direct reference to the passage (ie. "what is this passage about?").
    
Here is the passage:

PASSAGE:
{text_chunk}

Think step-by-step and reason through the content of the passage to hypothesize potential question-answer pairs and then respond. This of at least 5 question-answer pairs but more, if possible.

Respond in this format:
<think> You reasoning here... </think>
<qa-pair> [question here...] | [answer here...] </qa-pair>
<qa-pair> [question here...] | [answer here...] </qa-pair>
...
"""
    return prompt


def parse_response(response: str) -> dict:
    think, qa_pairs = response.split("</think>")[1].strip()
    qa_pairs = [pair.split("<qa-pair>")[1].strip() for pair in qa_pairs.split("</qa-pair>")]
    qa_pairs = [{"question": pair.split("|")[0].strip(), "answer": pair.split("|")[1].strip()} for pair in qa_pairs]
    return qa_pairs


# 3. Generate QA dataset
def generate_dataset():
    dataset = []
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
                qa_pairs = generate_qa_pairs_from_text(text)
                dataset.extend(qa_pairs)
    return dataset

if __name__ == "__main__":
    logging.info("Starting the QA dataset generation process")
    extract_text_from_pdfs()
    try:
        qa_dataset = asyncio.run(async_generate_dataset())

        # Optional: Save to JSON
        with open("surgical_qa_dataset.json", "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f)

        logging.info(f"Generated {len(qa_dataset)} QA pairs. Saved to surgical_qa_dataset.json")
        print(f"Generated {len(qa_dataset)} QA pairs.")
    except Exception as e:
        logging.error(f"An error occurred during dataset generation: {str(e)}")