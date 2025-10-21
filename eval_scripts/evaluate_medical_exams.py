import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import json
import os
import glob
import pandas as pd
import asyncio
import argparse
from dotenv import load_dotenv
from utils.index_w_faiss import FaissReader
import asyncio
import functools
import time
import logging
from utils.llms import init_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def to_thread(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    return wrapper

# Rate limiting constants
MAX_CALLS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60  # seconds

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.semaphore = asyncio.Semaphore(max_calls)

    async def acquire(self):
        await self.semaphore.acquire()
        
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            await asyncio.sleep(self.period - (now - self.calls[0]))
        
        self.calls.append(time.time())

    def release(self):
        self.semaphore.release()

rate_limiter = RateLimiter(MAX_CALLS_PER_MINUTE, RATE_LIMIT_PERIOD)

async def rate_limited_call(func, *args, **kwargs):
    await rate_limiter.acquire()
    try:
        return await func(*args, **kwargs)
    finally:
        rate_limiter.release()

async def answer_question(question, llm, use_cot=True, use_rag=False):
    logging.info(f"Generating answer for question: {question[:50]}...")

    use_context_instruction = ""
    context_string = ""
    if use_rag:
        faiss_reader = FaissReader("surgical_faiss_index")
        retrieved_docs = faiss_reader.search(question, k=5)
        context_string = "\n----\n## Relevant Context ##\n" + retrieved_docs + "\n----\n\n"
        use_context_instruction = " The following context may be helpful in answering the question."
    
    if use_cot:
        prompt = f"""You are a medical expert. Please answer the following question based on your medical knowledge.{use_context_instruction}

Question: {question}
{context_string}
Think step-by-step and provide a detailed reasoning process to arrive at your answer. Include at least 3 steps in your reasoning, but more as needed.

Your final answer must be : A / B / C / D (there may be more than 1 right answer)

Answer with a single letter when you think there is only one clear right answer (e.g. "A" or "B" or "C" or "D"). If there are multiple possible answers, answer with "A, B" or "A, B, C" or "A, B, C, D".

Respond in the following format:

<think> Your reasoning here... </think>
<answer> A / B / C / D </answer>
"""
    else:
        prompt = f"""You are a medical expert. Please answer the following question based on your medical knowledge.

Question: {question}

Your answer must be : A / B / C / D (there may be more than 1 right answer)

Answer with a single letter when you think there is only one clear right answer (e.g. "A" or "B" or "C" or "D"). If there are multiple possible answers, answer with "A, B" or "A, B, C" or "A, B, C, D".

Answer: """
    
    try:
        response = await rate_limited_call(to_thread(llm.invoke), prompt)
        
        if use_cot:
            try:
                thinking = response.content.split('<think>')[1].split('</think>')[0].strip()
                answer = response.content.split('<answer>')[1].split('</answer>')[0].strip().lower()
            except IndexError:
                # Fallback if the format is not followed
                thinking = response.content
                answer_text = response.content.lower()
                answer = organize_answer(answer_text)
        else:
            thinking = ""
            answer = organize_answer(response.content.strip().lower())
        
        logging.info(f"Answer generated for question: {question[:50]}...")
        logging.info(f"Answer: {answer}")
        return answer, thinking
    except Exception as e:
        logging.error(f"Error generating answer for question: {question[:50]}...")
        logging.error(str(e))
        return "maybe", "Could not generate answer for question."

def organize_answer(answer_text: str) -> str:
    answer_text = answer_text.lower()
    answer = ""
    if "a" in answer_text:
        answer += "a"
    if "b" in answer_text:
        answer += "b"
    if "c" in answer_text:
        answer += "c"
    if "d" in answer_text:
        answer += "d"
    return answer

def evaluate_answer(generated_answer, known_answer):
    """Simple string matching evaluation for yes/no/maybe answers"""
    generated_clean = organize_answer(generated_answer)
    known_clean = organize_answer(known_answer)
    
    is_correct = generated_clean == known_clean
    
    evaluation = f"Generated: '{generated_clean}', Expected: '{known_clean}', Match: {is_correct}"
    
    return is_correct, evaluation

async def process_question(item, results_file, llm, use_cot=True, use_rag=False):
    question = item['question']
    known_answer = item['answer']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Generate an answer
    generated_answer, CoT = await answer_question(question, llm, use_cot, use_rag)
    
    # Evaluate the answer
    is_correct, evaluation = evaluate_answer(generated_answer, known_answer)
    
    result = {
        'question': question,
        'known_answer': known_answer,
        'generated_answer': generated_answer,
        'CoT': CoT if use_cot else None,
        'is_correct': is_correct,
        'evaluation': evaluation,
        'used_cot': use_cot,
        'question_category': item['category']
    }

    # Append the result to the JSON file
    async with asyncio.Lock():
        with open(results_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    logging.info(f"Question processed and result saved: {question[:50]}...")
    return result

async def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM on PubMedQA dataset')
    parser.add_argument('--llm', type=str, default='azure-gpt4', 
                       help='LLM model to use (e.g., azure-gpt4, azure-gpt35, together-llama33)')
    parser.add_argument('--num_questions', type=int, default=None,
                       help='Number of questions to evaluate (default: all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file name (default: auto-generated based on model)')
    parser.add_argument('--no_cot', action='store_true',
                       help='Do not use Chain of Thought reasoning (default: use CoT)')
    parser.add_argument('--use_rag', action='store_true',
                       help='Use RAG (default: do not use RAG)')
    
    args = parser.parse_args()
    
    use_cot = not args.no_cot
    use_rag = args.use_rag
    
    logging.info("Starting the Medical Exam evaluation process...")
    logging.info(f"Using Chain of Thought: {use_cot}")
    logging.info(f"Using RAG: {use_rag}")
    
    # Initialize the LLM
    llm = init_llm(args.llm)
    logging.info(f"Initialized LLM: {args.llm}")
    
    # Load the Medical Exam dataset (pqa_labeled subset only)
    logging.info("Loading Medical Exam dataset...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Parent directory: {parent_dir}")
    dataset_files = glob.glob(os.path.join(parent_dir, 'medical_exams', '*.csv'))
    logging.info(f"Dataset files: {dataset_files}")
    all_dfs = []
    for file in dataset_files:
        df = pd.read_csv(file)
        df['category'] = os.path.basename(file).split('.csv')[0]
        all_dfs.append(df)
    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset_list = dataset.to_dict('records')
    
    # Convert to list and optionally limit number of questions
    if args.num_questions:
        dataset_list = dataset_list[:args.num_questions]
    
    logging.info(f"Loaded {len(dataset_list)} questions from Medical exam datasets")
    
    # Set up results file
    if args.output_file:
        results_file = args.output_file
    else:
        model_name = args.llm.replace('-', '_')
        cot_suffix = 'cot' if use_cot else 'no_cot'
        results_file = f'medical_exams_{cot_suffix}_results_{model_name}.json'
    
    # Initialize the results file
    with open(results_file, 'w') as f:
        json.dump([], f)

    logging.info(f"Processing {len(dataset_list)} questions...")

    # Process questions concurrently
    tasks = [process_question(item, results_file, llm, use_cot, use_rag) for item in dataset_list]
    results = await asyncio.gather(*tasks)

    # Calculate accuracy
    accuracy = sum(1 for result in results if result['is_correct']) / len(results)

    logging.info(f"Evaluation completed. Overall Accuracy: {accuracy:.2%}")
    
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())