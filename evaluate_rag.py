import json
from utils.agents import orchestrator, evaluate_answer, DeRetSynState
import os
from dotenv import load_dotenv
import multiprocessing
from tqdm import tqdm

# Load environment variables
load_dotenv()

def load_qa_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_question(qa_pair):
    question = qa_pair['question']
    known_answer = qa_pair['answer']

    # Initialize the state
    state = DeRetSynState(
        original_question=question,
        model=os.getenv('TOGETHER-LLAMA31'),
        api_key=os.getenv('TOGETHER_API_KEY'),
        base_url="https://api.together.xyz/v1/",
        collection_name="surgical_information"
    )

    # Run the orchestrator
    for step in orchestrator(state):
        if step['step'] == 'final':
            final_state = step['state']
            break

    # Evaluate the answer
    is_correct = evaluate_answer(final_state, known_answer)

    return {
        'question': question,
        'cot': final_state['cot_for_answer'],
        'rag_answer': final_state['final_answer'],
        'known_answer': known_answer,
        'is_correct': is_correct
    }

def run_evaluation(qa_dataset, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_question, qa_dataset), total=len(qa_dataset)))

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
    # Load the QA dataset
    qa_dataset = load_qa_dataset('surgical_qa_dataset.json')

    # Set the number of processes to use
    num_processes = min(1, int(multiprocessing.cpu_count()/2.0))  # Use half of all available CPU cores

    print(f"Starting evaluation with {num_processes} processes...")
    results = run_evaluation(qa_dataset, num_processes)

    print_results(results)

    # Save the evaluation results to a file
    with open('surgical_qa_dataset_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)