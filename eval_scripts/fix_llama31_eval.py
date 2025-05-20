import json
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import asyncio
import functools
import time
import logging

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

# Initialize the AzureChatOpenAI instance
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    model_name="gpt-4o",
    temperature=0.7
)

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
            wait_time = self.period - (now - self.calls[0])
            logging.info(f"Rate limit reached. Waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
        
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

async def evaluate_answer(question, generated_answer, known_answer):
    logging.info(f"Evaluating answer for question: {question[:50]}...")
    prompt = f"""You are a medical reasoning engine that compares two answers to a given question to determine whether the answers are the same. Here is the question and the two answers:

Question:
{question}

Answer 1 (Known Answer):
{known_answer}

Answer 2 (Generated Answer):
{generated_answer}

Think step-by-step and provide a detailed reasoning process that compares the two answers given the context of the question. Include at least 3 steps in your reasoning, but more as needed.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> TRUE if the answers are similar, FALSE otherwise... </answer>
"""
    try:
        response = await rate_limited_call(to_thread(llm.invoke), prompt)
        evaluation = response.content.strip()
        thinking = evaluation.split('<think>')[1].split('</think>')[0].strip()
        is_correct = 'true' in evaluation.lower().split('<answer>')[-1].split('</answer>')[0].strip()
        logging.info(f"Evaluation completed for question: {question[:50]}...")
        return is_correct, thinking
    except Exception as e:
        logging.error(f"Error evaluating answer for question: {question[:50]}. Error: {str(e)}")
        return False, "Could not evaluate answer for question."

async def process_question(item, results_file):
    question = item['question']
    known_answer = item['known_answer']
    rag_answer = item['rag_answer']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Evaluate the answer
    is_correct, evaluation = await evaluate_answer(question, rag_answer, known_answer)
    
    item['is_correct'] = is_correct
    item['evaluation'] = evaluation
    
    # Append the result to the JSON file
    async with asyncio.Lock():
        with open(results_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(item)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    logging.info(f"Question processed and result saved: {question[:50]}...")
    return item

async def main():
    logging.info("Starting the evaluation process...")
    
    # Load the dataset
    with open('surgical_qa_dataset_evaluation_results.json', 'r') as f:
        dataset = json.load(f)

    results_file = 'surgical_qa_dataset_evaluation_results_v2.json'

    # Initialize the results file
    with open(results_file, 'w') as f:
        json.dump([], f)

    logging.info(f"Processing {len(dataset)} questions...")

    # Process questions concurrently
    tasks = [process_question(item, results_file) for item in dataset]
    results = await asyncio.gather(*tasks)

    # Calculate accuracy
    accuracy = sum(1 for result in results if result['is_correct']) / len(results)

    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2%}")
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())