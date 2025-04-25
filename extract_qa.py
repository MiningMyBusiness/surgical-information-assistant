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
    
Think step-by-step and reason through the content of the passage to hypothesize potential question-answer pairs and then respond. Think of at least 8 question-answer pairs but more, if possible.

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
Okay, so I need to generate question-answer pairs based on the given medical passage. The questions should be answerable from the passage and understandable on their own. Let me read through the passage carefully to understand the key points.
The passage is about transvesical prostatectomy. It mentions that the goal is to remove the hypertrophic transitional zone of the prostate to reduce urinary obstruction. It's not for malignant lesions. Before any investigation, they do a history and physical exam, including a digital rectal exam. Enlarged lateral lobes can be felt, and the rectal mucosa should move freely. If there's nodularity or fixation, it might indicate cancer, so a biopsy is needed before admission.
Alright, so the main topics here are the purpose of the procedure, what it's not used for, the pre-procedure assessments, the physical exam findings, and when to suspect cancer.
I should think of questions that cover these areas. Let's start with the purpose. A question like, "What is the primary goal of transvesical prostatectomy?" The answer would be about removing the hypertrophic zone to relieve obstruction.
Next, since it's not for malignant lesions, a question could be, "Is transvesical prostatectomy used to treat malignant lesions of the prostate?" The answer would be no.
Then, the pre-procedure steps. Maybe, "What assessments are typically performed before transvesical prostatectomy?" The answer would include history, physical exam, and digital rectal exam.
For the physical exam findings, a question could be, "What is a normal finding during the digital rectal exam in a patient undergoing transvesical prostatectomy?" The answer would mention enlarged lateral lobes and freely moving rectal mucosa.
When to suspect cancer: "What signs during a digital rectal exam may indicate the need for a core needle biopsy?" The answer would be nodularity or fixation of the rectal mucosa.
I can also think about the implications of the findings. For example, "What should be done if nodularity or fixation is detected during the digital rectal exam?" The answer would be to perform a biopsy before proceeding.
Another question could be about the importance of the digital rectal exam in this context. "Why is a digital rectal exam important before transvesical prostatectomy?" The answer would be to assess for signs of cancer.
I should make sure each question is clear and can be answered without directly referencing the passage, just the information in it. Let me see if I can come up with more. Maybe about the specific zones of the prostate involved. "Which part of the prostate is targeted in transvesical prostatectomy?" The answer is the transitional zone.
Also, the procedure's focus: "What is the focus of transvesical prostatectomy in treating urinary obstruction?" The answer is removing the hypertrophic transitional zone.
I think that's a good set. I have eight pairs, but maybe I can find a couple more. Perhaps about the consequences of not doing the biopsy. "What is the consequence of not performing a core needle biopsy when suspicious findings are present?" The answer would be proceeding without knowing if cancer is present, which could be risky.
Wait, the passage doesn't explicitly state the consequences, so maybe that's beyond the given info. I should stick to what's in the passage.
Another angle: the patient's history. "What is included in the patient's history before transvesical prostatectomy?" The answer would be functional status inquiry.
I think that's a solid list. I'll make sure each question is clear and each answer is accurate based on the passage.
</think>
<qa-pair> What is the primary goal of transvesical prostatectomy? | The primary goal is to remove the hypertrophic transitional zone of the prostate to reduce urinary obstruction. </qa-pair>
<qa-pair> Is transvesical prostatectomy used to treat malignant lesions of the prostate? | No, it is not a surgical procedure for malignant lesions. </qa-pair>
<qa-pair> What assessments are typically performed before transvesical prostatectomy? | Appropriate history, including functional status inquiry, and physical examination are done prior to any investigation. </qa-pair>
<qa-pair> What is a normal finding during the digital rectal exam in a patient undergoing transvesical prostatectomy? | Enlarged lateral lobes can be easily felt, and the rectal mucosa should move freely over the prostate. </qa-pair>
<qa-pair> What signs during a digital rectal exam may indicate the need for a core needle biopsy? | Nodularity or fixation of the rectal mucosa may indicate the possibility of prostatic cancer. </qa-pair>
<qa-pair> What should be done if nodularity or fixation is detected during the digital rectal exam? | A core needle biopsy should be performed before proceeding with the admission. </qa-pair>
<qa-pair> Why is a digital rectal exam important before transvesical prostatectomy? | It is important to assess for signs of prostatic cancer, such as nodularity or fixation of the rectal mucosa. </qa-pair>
<qa-pair> Which part of the prostate is targeted in transvesical prostatectomy? | The transitional zone of the prostate is targeted. </qa-pair>
<qa-pair> What is the focus of transvesical prostatectomy in treating urinary obstruction? | The focus is on removing the hypertrophic transitional zone of the prostate. </qa-pair>
<qa-pair> What is included in the patient's history before transvesical prostatectomy? | The patient's history includes a functional status inquiry. </qa-pair>



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
Okay, so I need to generate question-answer pairs based on the given medical passage. The passage is about subcostal incisions and their uses in various surgical procedures. Let me read through it again to make sure I understand the key points.
The passage starts by saying that the subcostal incision is most useful when the pathology is on one side of the midline. It then gives examples for the left and right sides. On the left, examples include adrenalectomy, splenectomy, and trans-abdominal nephrectomy. On the right, it mentions cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy. It also talks about a bilateral subcostal incision providing excellent access for major hepatic resections and some surgeons using a central subcostal incision for esophagectomy.
Alright, so I need to create questions that are answerable from this passage but are understandable on their own. Each question should have enough detail to retrieve the relevant information in the future.
First, I can think about the main uses of the subcostal incision. So a question like, "What is the subcostal incision most useful for?" The answer would be situations where the pathology is on one side of the midline.
Next, looking at the examples on each side. I can create questions asking for examples on the left and right sides. For example, "What are examples of surgical procedures where a left-sided subcostal incision is used?" The answer would list adrenalectomy, splenectomy, and trans-abdominal nephrectomy.
Similarly, for the right side, the question would be, "What are examples of surgical procedures where a right-sided subcostal incision is used?" The answer includes cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy.
I should also consider the bilateral incision. So a question like, "What type of subcostal incision provides excellent access to all of the upper abdomen?" The answer is the bilateral subcostal incision, which is used for major hepatic resections.
Another point is the central subcostal incision for esophagectomy. So the question could be, "Why do some surgeons prefer a central subcostal incision?" The answer is for the abdominal part of esophagectomy.
I can also think about specific procedures and their relation to the incision. For example, "Which surgical procedures are commonly performed using a subcostal incision on the left side?" The answer would again be adrenalectomy, splenectomy, and trans-abdominal nephrectomy.
Similarly, for the right side, "Which surgical procedures are commonly performed using a subcostal incision on the right side?" The answer includes cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy.
I should also consider the application in major surgeries. So, "What type of subcostal incision is used for major hepatic resections?" The answer is the bilateral subcostal incision.
Lastly, the passage mentions that some surgeons use a central subcostal incision for esophagectomy. So the question could be, "In which surgical procedure is a central subcostal incision sometimes preferred?" The answer is esophagectomy, specifically the abdominal part.
I think that covers the main points. I should make sure each question is clear and can be answered without directly referencing the passage, just the information it contains. Also, each answer should be concise and accurate based on the passage.
</think>
<qa-pair> What is the subcostal incision most useful for? | The subcostal incision is most useful for situations where the location of the pathology is to one side of the midline. </qa-pair>
<qa-pair> What are examples of surgical procedures where a left-sided subcostal incision is used? | Examples include adrenalectomy, splenectomy, and trans-abdominal nephrectomy. </qa-pair>
<qa-pair> What are examples of surgical procedures where a right-sided subcostal incision is used? | Examples include cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy. </qa-pair>
<qa-pair> What type of subcostal incision provides excellent access to all of the upper abdomen? | The bilateral subcostal incision provides excellent access to all of the upper abdomen. </qa-pair>
<qa-pair> Why do some surgeons prefer a central subcostal incision? | Some surgeons prefer a central subcostal incision for the abdominal part of esophagectomy. </qa-pair>
<qa-pair> Which surgical procedures are commonly performed using a subcostal incision on the left side? | Adrenalectomy, splenectomy, and trans-abdominal nephrectomy are commonly performed using a left-sided subcostal incision. </qa-pair>
<qa-pair> Which surgical procedures are commonly performed using a subcostal incision on the right side? | Cholecystectomy, limited hepatic resections, adrenalectomy, and nephrectomy are commonly performed using a right-sided subcostal incision. </qa-pair>
<qa-pair> What type of subcostal incision is used for major hepatic resections? | The bilateral subcostal incision is used for major hepatic resections. </qa-pair>
<qa-pair> In which surgical procedure is a central subcostal incision sometimes preferred? | A central subcostal incision is sometimes preferred for the abdominal part of esophagectomy. </qa-pair>



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