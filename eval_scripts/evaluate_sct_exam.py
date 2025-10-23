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

faiss_reader = FaissReader("surgical_faiss_index")

async def answer_question(scenario, hypothesis, additional_info, llm, use_cot=True, use_rag=False):
    logging.info(f"Generating answer for question: {scenario[:50]}...")

    use_context_instruction = ""
    context_string = ""
    if use_rag:
        search_string = f"{scenario}\n{hypothesis}\n{additional_info}"
        retrieved_docs = faiss_reader.search(search_string, k=5)
        context_string = "\n----\n## Relevant Context\n" + retrieved_docs + "\n----\n\n"
        use_context_instruction = " The following context may be helpful."
    
    if use_cot:
        prompt = f"""You are taking a Script Concordance Test, which evaluates your understanding of medical knowledge.

In this test, you will evaluate how new information impacts a specific hypothesis. Use the following scale to rate the impact:

-2: Strongly decreases the likelihood of the hypothesis
-1: Slightly decreases the likelihood of the hypothesis
0: No effect on the likelihood of the hypothesis
+1: Slightly increases the likelihood of the hypothesis
+2: Strongly increases the likelihood of the hypothesis

{use_context_instruction}{context_string}

## Scenario
{scenario}

## Hypothesis
{hypothesis}

## Additional Information
{additional_info}

Think step-by-step and provide a detailed reasoning process to arrive at your answer. Include at least 3 steps in your reasoning, but more as needed.

Your final answer must be -2 or -1 or 0 or +1 or +2. This is an exam, and you are required to provide a valid answer.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> -2 or -1 or 0 or +1 or +2 </answer>
"""
    else:
        prompt = f"""You are taking a Script Concordance Test, which evaluates your understanding of medical knowledge.

In this test, you will evaluate how new information impacts a specific hypothesis. Use the following scale to rate the impact:

-2: Strongly decreases the likelihood of the hypothesis
-1: Slightly decreases the likelihood of the hypothesis
0: No effect on the likelihood of the hypothesis
+1: Slightly increases the likelihood of the hypothesis
+2: Strongly increases the likelihood of the hypothesis

## Scenario
{scenario}

## Hypothesis
{hypothesis}

## Additional Information
{additional_info}

Your answer must be : -2 or -1 or 0 or +1 or +2. Respond only with the answer.

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
        
        logging.info(f"Answer generated for question: {scenario[:50]}...")
        logging.info(f"Answer: {answer}")
        return answer, thinking
    except Exception as e:
        logging.error(f"Error generating answer for question: {scenario[:50]}...")
        logging.error(str(e))
        return "maybe", "Could not generate answer for question."

def organize_answer(answer_text: str) -> str:
    answer_text = answer_text.lower()
    if answer_text in ["-2", "-1", "0", "+1", "+2"]:
        return answer_text
    else:
        if "-2" in answer_text:
            return "-2"
        if "-1" in answer_text:
            return "-1"
        if "0" in answer_text:
            return "0"
        if "+1" in answer_text:
            return "+1"
        if "+2" in answer_text:
            return "+2"
        else:
            return "0"

def evaluate_answer(generated_answer: str, val_key: dict) -> float:
    """Simple string matching evaluation for yes/no/maybe answers"""
    generated_clean = organize_answer(generated_answer)
    points = val_key.get(generated_clean, 0)
    return points

async def process_question(item, results_file, llm, use_cot=True, use_rag=False):
    scenario = item['scenario']
    hypothesis = item['hypothesis']
    additional_info = item['additional_info']
    val_key = {
        "-2": item["-2"],
        "-1": item["-1"],
        "0": item["0"],
        "+1": item["1"],
        "+2": item["2"],
    }
    
    logging.info(f"Processing question: {scenario[:50]}...")
    
    # Generate an answer
    generated_answer, CoT = await answer_question(scenario, hypothesis, additional_info, llm, use_cot, use_rag)
    
    # Evaluate the answer
    points = evaluate_answer(generated_answer, val_key)
    
    result = {
        'scenario': scenario,
        'hypothesis': hypothesis,
        'additional_info': additional_info,
        'generated_answer': generated_answer,
        'CoT': CoT if use_cot else None,
        'is_correct': points,
        'used_cot': use_cot,
        'category': item['source']
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

    logging.info(f"Question processed and result saved: {scenario[:50]}...")
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
    
    logging.info("Starting the SCT evaluation process...")
    logging.info(f"Using Chain of Thought: {use_cot}")
    logging.info(f"Using RAG: {use_rag}")
    
    # Initialize the LLM
    llm = init_llm(args.llm)
    logging.info(f"Initialized LLM: {args.llm}")
    
    # Load the SCT dataset (pqa_labeled subset only)
    logging.info("Loading SCT dataset...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Parent directory: {parent_dir}")
    dataset_file = os.path.join(parent_dir, 'sct_data', 'sct_cleaned_full.csv')
    logging.info(f"Dataset file: {dataset_file}")
    df = pd.read_csv(dataset_file)
    df.rename(columns={'sct_stem': 'scenario', 'question': 'hypothesis'}, inplace=True)

    # Normalize the score columns
    score_columns = ['-2', '-1', '0', '1', '2']
    # Ensure score columns are numeric, coercing errors to NaN
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate the maximum value per row across the score columns
    row_max = df[score_columns].max(axis=1)

    # Avoid division by zero. If row_max is 0, the values in that row will not be changed.
    # The apply function will only be executed for rows where row_max > 0.
    df[score_columns] = df.apply(
        lambda row: row[score_columns] / row_max[row.name] if row_max[row.name] > 0 else row[score_columns],
        axis=1
    )

    dataset_list = df.to_dict('records')
    
    # Convert to list and optionally limit number of questions
    if args.num_questions:
        dataset_list = dataset_list[:args.num_questions]
    
    logging.info(f"Loaded {len(dataset_list)} questions from SCT datasets")
    
    # Set up results file
    if args.output_file:
        results_file = args.output_file
    else:
        model_name = args.llm.replace('-', '_')
        cot_suffix = 'cot' if use_cot else 'no_cot'
        rag_suffix = 'rag' if use_rag else 'no_rag'
        results_file = f'sct_{cot_suffix}_{rag_suffix}_results_{model_name}.json'
    
    # Initialize the results file
    with open(results_file, 'w') as f:
        json.dump([], f)

    logging.info(f"Processing {len(dataset_list)} questions...")

    # Process questions concurrently
    tasks = [process_question(item, results_file, llm, use_cot, use_rag) for item in dataset_list]
    results = await asyncio.gather(*tasks)

    # Calculate accuracy
    accuracy = sum(result['is_correct'] for result in results) / len(results)

    logging.info(f"Evaluation completed. Overall Accuracy: {accuracy:.2%}")
    
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())