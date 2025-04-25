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
import functools
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download("punkt")
nltk.download('punkt_tab')

def parse_args():
    if len(sys.argv) > 1:
        return sys.argv[1].lower() == 'async'
    return False  # Default to synchronous if no argument is provided

# Rate limiting constants
MAX_REQUESTS_PER_MINUTE = 250
SEMAPHORE_VALUE = 50  # Allow up to 50 concurrent requests
TIMEOUT_SECONDS = 60  # Timeout for each LLM request
USE_ASYNC = parse_args()

# Folder with PDFs
pdf_folder = "vumc_pdfs"
text_folder = "pdf_texts"
os.makedirs(text_folder, exist_ok=True)

def to_thread(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    return wrapper

start_time = [time.time()]


async def async_generate_qa_pair(chunk: str) -> List[Dict[str, str]]:
    llm = init_llm('azure-gpt35')
    prompt = get_prompt(chunk)
    logging.debug(f"Generating QA pair for chunk: {chunk[:50]}...")
    try:
        response = await to_thread(llm.invoke)(prompt)
        qa_pairs = parse_response(response.content)
        await to_thread(append_to_json)(qa_pairs)  # Append to JSON after each LLM call
        return qa_pairs
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
async def generate_qa_pairs_from_text(text: str, semaphore: asyncio.Semaphore, chunk_size: int = 10) -> List[Dict[str, str]]:
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
    prompt = f"""You are a medical reasoning engine. Given a medical or medically-related text passage, you must generate question-answer pairs. The questions MUST be answerable by the information in the passge but understandable on its own without needing direct reference to the passage (ie. "what is this passage about?" is NOT a valid question).
The questions should have enough information to be able to retrieve the relevant passages in the future to help answer the question. 
    
Think step-by-step and reason through the content of the passage to hypothesize potential question-answer pairs and then respond. Think of at least 4 question-answer pairs but more, if possible.

Respond in this format:
<think> You reasoning here... </think>
<qa-pair> [question here...] | [answer here...] </qa-pair>
<qa-pair> [question here...] | [answer here...] </qa-pair>

Here are some examples:

PASSAGE:
The goal of transvesical prostatectomy is to 
remove the hypertrophic transitional zone of the 
prostate to reduce urinary obstruction.  It is not a 
surgical procedure for malignant lesions. 
Appropriate history (including functional 
status inquiry) and physical examination are done 
prior to any investigation. Physical exam always 
includes digital rectal exam. Enlarged lateral lobes 
can be easily felt and the rectal mucosa should move 
freely over the prostate. Any nodularity or fixation of 
the rectal mucosa should alert you to the possibility 
of prostatic cancer and core needle biopsy should be 
done before proceeding with the admission.

RESPONSE:
<think>
Okay, so I need to generate question-answer pairs based on the given medical passage. The questions should be answerable by the passage and understandable on their own. Let me read through the passage carefully.
The passage talks about transvesical prostatectomy, its goal, what it's not used for, pre-procedure steps like history and physical exams, and specific exam findings that might indicate cancer.
First, I can think of a question about the purpose of transvesical prostatectomy. The answer is in the first sentence. Then, another question about what the procedure isn't used for, which is mentioned in the second sentence.
Next, the passage mentions the importance of history and physical exams before investigations. That can be another question. The physical exam includes a digital rectal exam, so a question about that makes sense.
The passage also describes what can be felt during the exam, like enlarged lateral lobes. So a question about the physical findings could be useful. Lastly, it talks about signs that might indicate cancer, such as nodularity, so a question about when to do a biopsy would be relevant.
I should make sure each question is clear and doesn't require direct reference to the passage. Let me structure them accordingly.
</think>
<qa-pair> What is the primary goal of transvesical prostatectomy? | The primary goal of transvesical prostatectomy is to remove the hypertrophic transitional zone of the prostate to reduce urinary obstruction. </qa-pair>
<qa-pair> What is transvesical prostatectomy not used for? | Transvesical prostatectomy is not a surgical procedure for malignant lesions. </qa-pair>
<qa-pair> What is included in the pre-procedure evaluation for transvesical prostatectomy? | Appropriate history, including functional status inquiry, and physical examination are done prior to any investigation. </qa-pair>
<qa-pair> What is a key component of the physical exam before transvesical prostatectomy? | The physical exam always includes a digital rectal exam. </qa-pair>



PASSAGE:
The subcostal incision is most useful for 
situations where the location of the pathology is to 
one side of the midline. Examples on the left side 
include adrenalectomy, splenectomy and trans-
abdominal nephrectomy. Examples on the right side 
include cholecystectomy, limited hepatic resections, 
adrenalectomy and nephrectomy. A bilateral 
subcostal incision gives excellent access to all of the 
upper abdomen, such as for major hepatic resections. 
Some surgeons prefer a central subcostal incision for 
the abdominal part of esophagectomy. 

RESPONSE:
<think>
Okay, so I need to generate question-answer pairs based on the given medical passage. The questions should be answerable by the passage and understandable on their own without direct reference. Let me read the passage carefully.
The passage talks about the subcostal incision, where it's most useful, and gives examples for left and right side procedures. It also mentions a bilateral subcostal incision and its use, as well as a central subcostal incision for esophagectomy.
First, I should identify key points. The main topic is the subcostal incision. It's useful when the pathology is on one side of the midline. Left side examples include adrenalectomy, splenectomy, and trans-abdominal nephrectomy. Right side examples are cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy. A bilateral incision is good for major hepatic resections. Some surgeons use a central subcostal incision for esophagectomy.
Now, I need to think of questions that can be answered from this. Let's start with the general use of subcostal incision. Then, specific procedures on each side. Also, the bilateral and central uses.
I should make sure the questions are clear and self-contained. For example, "What is the subcostal incision most useful for?" That's a good start. The answer would be about situations where pathology is on one side of the midline.
Next, for left side examples. "What surgical procedures are commonly performed using a left subcostal incision?" The answer would list adrenalectomy, splenectomy, and trans-abdominal nephrectomy.
Similarly, for the right side: "What surgical procedures are commonly performed using a right subcostal incision?" Answer includes cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy.
Then, the bilateral incision: "What is a bilateral subcostal incision typically used for?" Answer is major hepatic resections.
Lastly, the central subcostal incision: "Why might a central subcostal incision be preferred by some surgeons?" Answer relates to the abdominal part of esophagectomy.
I think that's five good Q&A pairs. Each question is specific and can be answered without referring back to the passage. They cover the main points and some specifics, which should be helpful for future reference.
</think>
<qa-pair> What is the subcostal incision most useful for? | The subcostal incision is most useful for situations where the location of the pathology is to one side of the midline. </qa-pair>
<qa-pair> What surgical procedures are commonly performed using a left subcostal incision? | Adrenalectomy, splenectomy, and trans-abdominal nephrectomy are commonly performed using a left subcostal incision. </qa-pair>
<qa-pair> What surgical procedures are commonly performed using a right subcostal incision? | Cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy are commonly performed using a right subcostal incision. </qa-pair>
<qa-pair> What is a bilateral subcostal incision typically used for? | A bilateral subcostal incision is typically used for major hepatic resections. </qa-pair>
<qa-pair> Why might a central subcostal incision be preferred by some surgeons? | Some surgeons prefer a central subcostal incision for the abdominal part of esophagectomy. </qa-pair>



PASSAGE:
{text_chunk}

RESPONSE:
"""
    return prompt


def parse_response(response: str) -> dict:
    logging.debug(f"Parsing response: {response}")
    try:
        qa_pairs = response.split("</think>")[1].strip()
        qa_pairs = [pair.replace("<qa-pair>", "").strip() for pair in qa_pairs.split("</qa-pair>")]
        qa_pairs = [{"question": pair.split("|")[0].strip(), "answer": pair.split("|")[1].strip()} for pair in qa_pairs if '|' in pair]
        return qa_pairs
    except Exception as e:
        logging.error(f"Error parsing response: {str(e)}")
        return []


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

def serial_generate_qa_pair(chunk: str) -> List[Dict[str, str]]:
    llm = init_llm('azure-gpt35')
    prompt = get_prompt(chunk)
    logging.debug(f"Generating QA pair for chunk: {chunk[:50]}...")
    try:
        response = llm.invoke(prompt)
        qa_pairs = parse_response(response.content)
        append_to_json(qa_pairs)  # Append to JSON after each LLM call
        return qa_pairs
    except Exception as e:
        logging.error(f"Error occurred while generating QA pair: {str(e)}")
        return []

def serial_generate_qa_pairs_from_text(text: str, chunk_size: int = 5, return_chunks: bool = False) -> List[Dict[str, str]]:
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]
    logging.info(f"Generated {len(chunks)} chunks from text")

    if return_chunks:
        return chunks

    qa_pairs = []
    for chunk in chunks:
        qa_pairs.extend(serial_generate_qa_pair(chunk))

    logging.info(f"Generated {len(qa_pairs)} QA pairs from text")
    return qa_pairs

def serial_generate_dataset():
    dataset = []
    request_count = 0
    total_files = len([f for f in os.listdir(text_folder) if f.endswith(".txt")])
    processed_files = 0
    total_time = 0
    total_qa_pairs = 0
    overall_start_time = time.time()

    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            logging.info(f"Processing file: {filename}")
            try:
                with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                
                file_start_time = time.time()
                chunks = serial_generate_qa_pairs_from_text(text, return_chunks=True)
                
                for i, chunk in enumerate(chunks):
                    chunk_start_time = time.time()
                    qa_pairs = serial_generate_qa_pair(chunk)
                    chunk_end_time = time.time()
                    
                    chunk_time = chunk_end_time - chunk_start_time
                    total_time += chunk_time
                    total_qa_pairs += len(qa_pairs)
                    
                    dataset.extend(qa_pairs)
                    request_count += len(qa_pairs)
                    
                    # Calculate and display timing information
                    avg_time_per_pair = chunk_time / len(qa_pairs) if qa_pairs else 0
                    processed_chunks = i + 1
                    remaining_chunks = len(chunks) - processed_chunks
                    projected_remaining_time = (total_time / processed_chunks) * remaining_chunks if processed_chunks > 0 else 0
                    
                    logging.info(f"Generated {len(qa_pairs)} QA pairs for chunk {i+1}/{len(chunks)} in {filename}")
                    logging.info(f"Time taken for this chunk: {chunk_time:.2f} seconds")
                    logging.info(f"Average time per QA pair: {avg_time_per_pair:.2f} seconds")
                    logging.info(f"Projected remaining time for this file: {projected_remaining_time:.2f} seconds")
                    
                    # Check if we need to pause to respect rate limit
                    elapsed_time = time.time() - start_time[0]
                    if elapsed_time < 60 and request_count >= MAX_REQUESTS_PER_MINUTE:
                        sleep_time = 60 - elapsed_time
                        logging.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                        time.sleep(sleep_time)
                        start_time[0] = time.time()
                        request_count = 0
                
                processed_files += 1
                remaining_files = total_files - processed_files
                overall_elapsed_time = time.time() - overall_start_time
                overall_avg_time_per_file = overall_elapsed_time / processed_files if processed_files > 0 else 0
                overall_projected_remaining_time = overall_avg_time_per_file * remaining_files
                
                logging.info(f"Finished processing file: {filename}")
                logging.info(f"Processed {processed_files}/{total_files} files")
                logging.info(f"Overall elapsed time: {overall_elapsed_time:.2f} seconds")
                logging.info(f"Projected remaining time for all files: {overall_projected_remaining_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")

    overall_avg_time = total_time / total_qa_pairs if total_qa_pairs > 0 else 0
    logging.info(f"Total QA pairs generated: {len(dataset)}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")
    logging.info(f"Overall average time per QA pair: {overall_avg_time:.2f} seconds")
    return dataset


def append_to_json(qa_pairs, filename="surgical_qa_dataset.json"):
    try:
        # Read existing data
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/invalid, start with an empty list
        data = []
    
    # Append new QA pairs
    data.extend(qa_pairs)
    
    # Write updated data back to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Appended {len(qa_pairs)} QA pairs to {filename}")


if __name__ == "__main__":
    logging.info("Starting the QA dataset generation process")
    logging.info(f"Running in {'async' if USE_ASYNC else 'sync'} mode")
    extract_text_from_pdfs()
    try:
        if USE_ASYNC:
            qa_dataset = asyncio.run(async_generate_dataset())
        else:
            qa_dataset = serial_generate_dataset()

        logging.info(f"Generated {len(qa_dataset)} QA pairs. Saved incrementally to surgical_qa_dataset.json")
        print(f"Generated {len(qa_dataset)} QA pairs.")
    except Exception as e:
        logging.error(f"An error occurred during dataset generation: {str(e)}")
    finally:
        print("\nUsage:")
        print("  python extract_qa.py [mode]")
        print("    mode: 'async' for asynchronous execution, any other value or omit for synchronous execution")
        print(f"Current mode: {'async' if USE_ASYNC else 'sync'}")