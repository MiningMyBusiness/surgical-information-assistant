import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import json
import os
import asyncio
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
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
MAX_CALLS_PER_MINUTE = 50
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

async def answer_question(question, context, llm, use_context=True, use_cot=True):
    logging.info(f"Generating answer for question: {question[:50]}...")
    
    if use_cot:
        if use_context:
            prompt = f"""You are a medical expert. Please answer the following question based on the provided context.

Context:
{context}

Question: {question}

Think step-by-step and provide a detailed reasoning process to arrive at your answer. Include at least 3 steps in your reasoning, but more as needed.

Your final answer must be exactly one of: "yes", "no", or "maybe"

Respond in the following format:

<think> Your reasoning here... </think>
<answer> yes/no/maybe </answer>
"""
        else:
            prompt = f"""You are a medical expert. Please answer the following question based on your medical knowledge.

Question: {question}

Think step-by-step and provide a detailed reasoning process to arrive at your answer. Include at least 3 steps in your reasoning, but more as needed.

Your final answer must be exactly one of: "yes", "no", or "maybe"

Respond in the following format:

<think> Your reasoning here... </think>
<answer> yes/no/maybe </answer>
"""
    else:
        if use_context:
            prompt = f"""You are a medical expert. Please answer the following question based on the provided context.

Context:
{context}

Question: {question}

Your answer must be exactly one of: "yes", "no", or "maybe"

Answer: """
        else:
            prompt = f"""You are a medical expert. Please answer the following question based on your medical knowledge.

Question: {question}

Your answer must be exactly one of: "yes", "no", or "maybe"

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
                if 'yes' in answer_text and 'no' not in answer_text:
                    answer = 'yes'
                elif 'no' in answer_text and 'yes' not in answer_text:
                    answer = 'no'
                elif 'maybe' in answer_text:
                    answer = 'maybe'
                else:
                    answer = 'maybe'
        else:
            thinking = ""
            answer = response.content.strip().lower()
        
        # Ensure answer is one of the valid options
        if answer not in ['yes', 'no', 'maybe']:
            # Try to extract from the answer text
            answer_text = answer.lower()
            if 'yes' in answer_text and 'no' not in answer_text:
                answer = 'yes'
            elif 'no' in answer_text and 'yes' not in answer_text:
                answer = 'no'
            elif 'maybe' in answer_text:
                answer = 'maybe'
            else:
                answer = 'maybe'  # Default fallback
        
        logging.info(f"Answer generated for question: {question[:50]}...")
        logging.info(f"Answer: {answer}")
        return answer, thinking
    except Exception as e:
        logging.error(f"Error generating answer for question: {question[:50]}...")
        logging.error(str(e))
        return "maybe", "Could not generate answer for question."

def evaluate_answer(generated_answer, known_answer):
    """Simple string matching evaluation for yes/no/maybe answers"""
    generated_clean = generated_answer.lower().strip()
    known_clean = known_answer.lower().strip()
    
    is_correct = generated_clean == known_clean
    
    evaluation = f"Generated: '{generated_clean}', Expected: '{known_clean}', Match: {is_correct}"
    
    return is_correct, evaluation

async def process_question(item, results_file, llm, use_context=True, use_cot=True):
    question = item['question']
    context = item['context']
    known_answer = item['final_decision']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Generate an answer
    generated_answer, CoT = await answer_question(question, context, llm, use_context, use_cot)
    
    # Evaluate the answer
    is_correct, evaluation = evaluate_answer(generated_answer, known_answer)
    
    result = {
        'question': question,
        'context': context if use_context else None,
        'known_answer': known_answer,
        'generated_answer': generated_answer,
        'CoT': CoT if use_cot else None,
        'is_correct': is_correct,
        'evaluation': evaluation,
        'used_context': use_context,
        'used_cot': use_cot
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
    parser.add_argument('--no_context', action='store_true',
                       help='Do not use context when answering questions (default: use context)')
    parser.add_argument('--no_cot', action='store_true',
                       help='Do not use Chain of Thought reasoning (default: use CoT)')
    
    args = parser.parse_args()
    
    use_context = not args.no_context
    use_cot = not args.no_cot
    
    logging.info("Starting the PubMedQA evaluation process...")
    logging.info(f"Using context: {use_context}")
    logging.info(f"Using Chain of Thought: {use_cot}")
    
    # Initialize the LLM
    llm = init_llm(args.llm)
    logging.info(f"Initialized LLM: {args.llm}")
    
    # Load the PubMedQA dataset (pqa_labeled subset only)
    logging.info("Loading PubMedQA dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    
    # Convert to list and optionally limit number of questions
    dataset_list = list(dataset)
    if args.num_questions:
        dataset_list = dataset_list[:args.num_questions]
    
    logging.info(f"Loaded {len(dataset_list)} questions from PubMedQA")
    
    # Set up results file
    if args.output_file:
        results_file = args.output_file
    else:
        model_name = args.llm.replace('-', '_')
        context_suffix = 'with_context' if use_context else 'no_context'
        cot_suffix = 'cot' if use_cot else 'no_cot'
        results_file = f'pubmedqa_{cot_suffix}_results_{model_name}_{context_suffix}.json'
    
    # Initialize the results file
    with open(results_file, 'w') as f:
        json.dump([], f)

    logging.info(f"Processing {len(dataset_list)} questions...")

    # Process questions concurrently
    tasks = [process_question(item, results_file, llm, use_context, use_cot) for item in dataset_list]
    results = await asyncio.gather(*tasks)

    # Calculate accuracy
    accuracy = sum(1 for result in results if result['is_correct']) / len(results)
    
    # Calculate per-class accuracy
    yes_correct = sum(1 for result in results if result['known_answer'].lower() == 'yes' and result['is_correct'])
    no_correct = sum(1 for result in results if result['known_answer'].lower() == 'no' and result['is_correct'])
    maybe_correct = sum(1 for result in results if result['known_answer'].lower() == 'maybe' and result['is_correct'])
    
    yes_total = sum(1 for result in results if result['known_answer'].lower() == 'yes')
    no_total = sum(1 for result in results if result['known_answer'].lower() == 'no')
    maybe_total = sum(1 for result in results if result['known_answer'].lower() == 'maybe')

    logging.info(f"Evaluation completed. Overall Accuracy: {accuracy:.2%}")
    if yes_total > 0:
        logging.info(f"'Yes' Accuracy: {yes_correct/yes_total:.2%} ({yes_correct}/{yes_total})")
    if no_total > 0:
        logging.info(f"'No' Accuracy: {no_correct/no_total:.2%} ({no_correct}/{no_total})")
    if maybe_total > 0:
        logging.info(f"'Maybe' Accuracy: {maybe_correct/maybe_total:.2%} ({maybe_correct}/{maybe_total})")
    
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())