import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from utils.agents import orchestrator_straight_run, DeRetSynState
from utils.index_w_faiss import FaissReader
from dotenv import load_dotenv
import asyncio
import multiprocessing
from tqdm import tqdm
import sys
import time
import logging
from utils.llms import init_llm

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rate limiting constants
MAX_CALLS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60  # seconds

def load_qa_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def load_eval_results(file_path: str="surgical_qa_dataset_evaluation_results_llama32-3b.json"):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.info(f"No evaluation results found at {file_path}")
        return []
    

def get_all_evaluated_questions(eval_results):
    return [q['question'] for q in eval_results]

ALL_EVAL_QUESTIONS_SO_FAR = get_all_evaluated_questions(load_eval_results())


# Initialize the AzureChatOpenAI instance
eval_llm = init_llm('azure-gpt35')

faiss_reader = FaissReader("surgical_faiss_index")

def append_to_json_file(result: dict, file_path: str="surgical_qa_dataset_evaluation_results_llama32-3b_2025oct_2.json"):
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


async def process_question_async(qa_pair):
    question = qa_pair['question']
    known_answer = qa_pair['answer']

    # Initialize the state
    state = DeRetSynState(
        original_question=question,
        model=os.getenv('TOGETHER_LLAMA32'),
        api_key=os.getenv('TOGETHER_API_KEY'),
        base_url=os.getenv('TOGETHER_URL'),
        faiss_index_path="surgical_faiss_index",
        verbose=True,
        iterations=0,
        wikipedia_results="",
        run_async=True,
        vectorstore=faiss_reader
    )

    # Run the orchestrator
    final_state = orchestrator_straight_run(state)

    # Evaluate the answer
    is_correct, thinking = await evaluate_answer(
        question=question,
        generated_answer=final_state['final_answer'],
        known_answer=known_answer
    )

    output = {
        'question': question,
        'document_context': final_state['answers'],
        'wikipedia_context': final_state['wikipedia_results'],
        'cot': final_state['cot_for_answer'],
        'rag_answer': final_state['final_answer'],
        'rag_confidence': final_state['final_confidence'],
        'known_answer': known_answer,
        'is_correct': is_correct,
        'eval_thinking': thinking
    }

    return output


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


def process_question(qa_pair):

    question = qa_pair['question']
    if question in ALL_EVAL_QUESTIONS_SO_FAR:
        logging.info(f"Skipping evaluation for question {question} (already evaluated)")
        return None
    known_answer = qa_pair['answer']

    # Initialize the state
    state = DeRetSynState(
        original_question=question,
        model=os.getenv('TOGETHER_LLAMA32'),
        api_key=os.getenv('TOGETHER_API_KEY'),
        base_url=os.getenv('TOGETHER_URL'),
        faiss_index_path="surgical_faiss_index",
        verbose=False,
        iterations=0,
        wikipedia_results="",
        run_async=True,
        vectorstore=faiss_reader
    )

    # Run the orchestrator
    try:
        for step in orchestrator(state):
            if step['step'] == 'final':
                final_state = step['state']
                break
        # Evaluate the answer
        is_correct, thinking = asyncio.run(evaluate_answer(
            question=question,
            generated_answer=final_state['final_answer'],
            known_answer=known_answer
        ))

        output = {
            'question': question,
            'document_context': final_state['answers'],
            'wikipedia_context': final_state['wikipedia_results'],
            'cot': final_state['cot_for_answer'],
            'rag_answer': final_state['final_answer'],
            'rag_confidence': final_state['final_confidence'],
            'known_answer': known_answer,
            'is_correct': is_correct,
            'eval_thinking': thinking
        }

        return output
    except Exception as e:
        logging.error(f"Error running orchestrator for question {question}: {str(e)}")
        return {
        'question': question,
        'document_context': None,
        'wikipedia_context': None,
        'cot': None,
        'rag_answer': None,
        'rag_confidence': None,
        'known_answer': known_answer,
        'is_correct': False,
        'eval_thinking': None
    }

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

async def evaluate_answer(question, generated_answer, known_answer):
    logging.info(f"Evaluating answer for question: {question[:50]}...")
    prompt = f"""You are a medical reasoning engine that compares a generated answer with a known answer to a given question to determine whether the generated answer is correct. Here is the question and the two answers:

Question:
{question}

Answer 1 (Known Answer):
{known_answer}

Answer 2 (Generated Answer):
{generated_answer}

Think step-by-step and provide a detailed reasoning process that compares the two answers given the context of the question. Include at least 3 steps in your reasoning, but more as needed.

Keep these criteria in mind:
1. The generated answer should contain the core information in the known answer that is directly relevant to the question
2. The generated answer can contain more detail but it should NOT contradict the known answer
3. The generated answer can be more or less concise than the known answer
4. The generated answer should answer the question

Respond in the following format:

<think> Your reasoning here... </think>
<answer> TRUE if the generated answer is correct, FALSE otherwise... </answer>
"""
    try:
        response = await rate_limited_call(to_thread(eval_llm.invoke), prompt)
        evaluation = response.content.strip()
        thinking = evaluation.split('<think>')[1].split('</think>')[0].strip()
        is_correct = 'true' in evaluation.lower().split('<answer>')[-1].split('</answer>')[0].strip()
        logging.info(f"Evaluation completed for question: {question[:50]}...")
        logging.info(f"Evalution result: {is_correct}")
        return is_correct, thinking
    except Exception as e:
        logging.error(f"Error evaluating answer for question: {question[:50]}...")
        logging.error(str(e))
        return False, "Could not evaluate answer for question."

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
    # grab asyn or sync run based on command line arguments
    is_async = len(sys.argv) > 1 and sys.argv[1] == "async"
    if is_async:
        print("Running evaluation asynchronously...")
    
    # grab num_processes from command line arguments
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Running evaluation with {num_processes} processes...")

    # Load the QA dataset
    qa_dataset = load_qa_dataset('surgical_qa_dataset_2025oct_2_cleaned.json')

    # Set the number of processes to use
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count()/1.5))  # Use half of all available CPU cores
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count() - 1)  # Limit to the number of available CPU cores

    if not is_async:
        print(f"Starting evaluation with {num_processes} processes...")
        results = run_evaluation(qa_dataset, num_processes)
    else:
        results = asyncio.run(run_evaluation_async(qa_dataset))

    print_results(results)

    # Save the evaluation results to a file
    with open('surgical_qa_dataset_evaluation_results_llama32-3b_2025oct_2.json', 'w') as f:
        json.dump(results, f, indent=4)