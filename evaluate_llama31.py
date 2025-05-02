import json
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
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
llm_4o = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    model_name="gpt-4o",
    temperature=0.7
)

llm_llama31 = ChatOpenAI(model=os.environ["TOGETHER_LLAMA31"],
                         api_key=os.environ["TOGETHER_API_KEY"],
                         base_url="https://api.together.xyz/v1/",
                         temperature=0.7)

# Rate limiting constants
MAX_CALLS_PER_MINUTE = 100
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

async def answer_question(question):
    logging.info(f"Generating answer for question: {question[:50]}...")
    prompt = f"""You are a medical expert. Please answer the following question:
Think step-by-step and provide a detailed reasoning process to arrive at your answer. Include at least 3 steps in your reasoning, but more as needed.

Respond in the following format:

<thinking> Your reasoning here... </thinking>
<answer> Your final answer here... </answer>

Question: {question}
"""
    
    try:
        response = await rate_limited_call(to_thread(llm_llama31.invoke), prompt)
        thinking = response.content.split('<thinking>')[1].split('</thinking>')[0].strip()
        answer = response.content.split('<answer>')[1].split('</answer>')[0].strip()
        logging.info(f"Answer generated for question: {question[:50]}...")
        logging.info(f"Answer first few words: {answer[:50]}...")
        return answer, thinking
    except Exception as e:
        logging.error(f"Error generating answer for question: {question[:50]}...")
        logging.error(str(e))
        return "Could not generate answer for question.", "Could not generate answer for question."

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

<thinking> Your reasoning here... </thinking>
<answer> TRUE if the answers are similar, FALSE otherwise... </answer>
"""
    try:
        response = await rate_limited_call(to_thread(llm_4o.invoke), prompt)
        evaluation = response.content.strip()
        thinking = evaluation.split('<thinking>')[1].split('</thinking>')[0].strip()
        is_correct = 'true' in evaluation.lower().split('<answer>')[-1].split('</answer>')[0].strip()
        logging.info(f"Evaluation completed for question: {question[:50]}...")
        logging.info(f"Evalution result: {is_correct}")
        return is_correct, thinking
    except Exception as e:
        logging.error(f"Error evaluating answer for question: {question[:50]}...")
        logging.error(str(e))
        return False, "Could not evaluate answer for question."

async def process_question(item, results_file):
    question = item['question']
    known_answer = item['answer']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Generate an answer
    generated_answer, CoT = await answer_question(question)
    
    # Evaluate the answer
    is_correct, evaluation = await evaluate_answer(question, generated_answer, known_answer)
    
    result = {
        'question': question,
        'known_answer': known_answer,
        'generated_answer': generated_answer,
        'CoT': CoT,
        'is_correct': is_correct,
        'evaluation': evaluation
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
    logging.info("Starting the evaluation process...")
    
    # Load the dataset
    with open('surgical_qa_dataset.json', 'r') as f:
        dataset = json.load(f)

    results_file = 'qa_results_llama31_wo_rag.json'

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