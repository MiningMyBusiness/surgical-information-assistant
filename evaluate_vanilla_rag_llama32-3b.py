import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio
import multiprocessing
from tqdm import tqdm
import sys
import time
import logging
from utils.index_w_faiss import FaissReader
from utils.agents import evaluate_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Rate limiting constants
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_PERIOD = 60  # seconds

def load_qa_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_eval_results(file_path: str="vanilla_rag_evaluation_results_llama32.json"):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.info(f"No evaluation results found at {file_path}")
        return []
    
def get_all_evaluated_questions(eval_results):
    return [q['question'] for q in eval_results]

ALL_EVAL_QUESTIONS_SO_FAR = get_all_evaluated_questions(load_eval_results())

# Initialize the LLM instances
rag_llm = ChatOpenAI(
    model=os.getenv('TOGETHER_LLAMA32'),
    api_key=os.getenv('TOGETHER_API_KEY'),
    base_url=os.getenv('TOGETHER_URL'),
    temperature=0.2
)

eval_llm = ChatOpenAI(
    model=os.getenv('TOGETHER_MISTRAL'),
    api_key=os.getenv('TOGETHER_API_KEY'),
    base_url=os.getenv('TOGETHER_URL'),
    temperature=0.7
)

# Initialize the FAISS reader
faiss_reader = FaissReader("surgical_faiss_index")

def append_to_json_file(result: dict, file_path: str="vanilla_rag_evaluation_results_llama32.json"):
    try:
        if not os.path.exists(file_path):
            logging.info(f"Creating new file: {file_path}")
            with open(file_path, 'w') as f:
                json.dump([], f)
        
        logging.info(f"Appending to file: {file_path}")
        with open(file_path, 'r+') as f:
            data = json.load(f)
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        logging.info(f"Successfully appended to file: {file_path}")
    except Exception as e:
        logging.error(f"Error appending to file {file_path}: {str(e)}")

def generate_answer_from_context(question, context_docs, llm):
    """Generate an answer using the LLM based on the retrieved context."""
    
    prompt = f"""You are a medical assistant specializing in surgical information. Use the following context to answer the question. 
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context_docs}

Question: {question}

Think step-by-step to reason through your answer and consider the relevant information from the context. Respond in the following format:
<think> Your reasoning here... </think>
<answer> The answer to the question... </answer>
"""
    
    response = llm.invoke(prompt).content.strip()
    
    # Extract the thinking and answer parts
    thinking = ""
    answer = ""
    
    if "<think>" in response and "</think>" in response:
        thinking = response.split("<think>")[1].split("</think>")[0].strip()
    
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        # If the format wasn't followed, use the whole response as the answer
        answer = response
        
    return answer, thinking, context_docs

def process_question(qa_pair):
    question = qa_pair['question']
    if question in ALL_EVAL_QUESTIONS_SO_FAR:
        logging.info(f"Skipping evaluation for question {question} (already evaluated)")
        return None
    
    known_answer = qa_pair['answer']

    try:
        # Retrieve relevant documents from FAISS
        retrieved_docs = faiss_reader.search(question, k=3)
        
        # Generate answer using the LLM
        answer, thinking, context = generate_answer_from_context(question, retrieved_docs, rag_llm)
        
        # Create a state-like object for evaluation
        state = {
            "original_question": question,
            "final_answer": answer,
            "cot_for_answer": thinking,
            "verbose": False
        }
        
        # Evaluate the answer
        is_correct = evaluate_answer(state, known_answer, eval_llm)

        output = {
            'question': question,
            'document_context': context,
            'cot': thinking,
            'rag_answer': answer,
            'known_answer': known_answer,
            'is_correct': is_correct
        }

        return output
    except Exception as e:
        logging.error(f"Error processing question {question}: {str(e)}")
        return {
            'question': question,
            'document_context': None,
            'cot': None,
            'rag_answer': None,
            'known_answer': known_answer,
            'is_correct': False
        }

async def process_question_async(qa_pair):
    question = qa_pair['question']
    known_answer = qa_pair['answer']

    try:
        # Retrieve relevant documents from FAISS
        retrieved_docs = faiss_reader.search(question, k=5)
        
        # Generate answer using the LLM
        answer, thinking, context = generate_answer_from_context(question, retrieved_docs, rag_llm)
        
        # Create a state-like object for evaluation
        state = {
            "original_question": question,
            "final_answer": answer,
            "cot_for_answer": thinking,
            "verbose": False
        }
        
        # Evaluate the answer
        is_correct = evaluate_answer(state, known_answer, eval_llm)

        output = {
            'question': question,
            'document_context': context,
            'cot': thinking,
            'rag_answer': answer,
            'known_answer': known_answer,
            'is_correct': is_correct
        }

        return output
    except Exception as e:
        logging.error(f"Error processing question {question}: {str(e)}")
        return {
            'question': question,
            'document_context': None,
            'cot': None,
            'rag_answer': None,
            'known_answer': known_answer,
            'is_correct': False
        }

async def run_evaluation_async(qa_dataset):
    semaphore = asyncio.Semaphore(MAX_CALLS_PER_MINUTE)
    start_time = time.time()
    calls_made = 0

    async def process_with_rate_limit(qa_pair):
        nonlocal start_time, calls_made

        async with semaphore:
            # Check if we need to reset the timer
            current_time = time.time()
            if current_time - start_time >= RATE_LIMIT_PERIOD:
                start_time = current_time
                calls_made = 0

            # If we've reached the limit, wait until the next period
            if calls_made >= MAX_CALLS_PER_MINUTE:
                wait_time = RATE_LIMIT_PERIOD - (current_time - start_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                start_time = time.time()
                calls_made = 0

            result = await process_question_async(qa_pair)
            calls_made += 1
            append_to_json_file(result)
            return result

    tasks = [process_with_rate_limit(qa_pair) for qa_pair in qa_dataset]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(qa_dataset)):
        result = await task
        results.append(result)

    return results

def run_evaluation(qa_dataset, num_processes):
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_question, qa_dataset), total=len(qa_dataset)):
                if result:
                    append_to_json_file(result)
                    results.append(result)
    else:
        results = []
        for qa_pair in tqdm(qa_dataset, total=len(qa_dataset)):
            result = process_question(qa_pair)
            if result:
                append_to_json_file(result)
                results.append(result)

    return results

def print_results(results):
    total_questions = len(results)
    correct_answers = sum(1 for result in results if result['is_correct'])

    print("\nEvaluation Results:")
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}:")
        print(f"Question: {result['question']}")
        print(f"RAG Answer: {result['rag_answer']}")
        print(f"Known Answer: {result['known_answer']}")
        print(f"Evaluation: {'Correct' if result['is_correct'] else 'Incorrect'}")

    accuracy = (correct_answers / total_questions) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print(f"Correct Answers: {correct_answers}/{total_questions}")

if __name__ == "__main__":
    # grab async or sync run based on command line arguments
    is_async = len(sys.argv) > 1 and sys.argv[1] == "async"
    if is_async:
        print("Running evaluation asynchronously...")
    
    # grab num_processes from command line arguments
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Load the QA dataset
    qa_dataset = load_qa_dataset('surgical_qa_dataset.json')

    # Set the number of processes to use
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count()/1.5))  # Use half of all available CPU cores
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count() - 1)  # Limit to the number of available CPU cores

    print(f"Running evaluation with {num_processes} processes...")

    if not is_async:
        print(f"Starting evaluation with {num_processes} processes...")
        results = run_evaluation(qa_dataset, num_processes)
    else:
        results = asyncio.run(run_evaluation_async(qa_dataset))

    print_results(results)

    # Save the evaluation results to a file
    with open('vanilla_rag_evaluation_results_llama32.json', 'w') as f:
        json.dump(results, f, indent=4)