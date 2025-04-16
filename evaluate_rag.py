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
    

def milvus_copy_per_process(num_processes: int, milvus_db_path: str="./milvus.db"):
    # Create a separate copy of milvus database for each process with random filename
    milvus_dbs = [f"milvus_{str(i).zfill(4)}.db" for i in range(num_processes)]
    for i, db_path in enumerate(milvus_dbs):
        os.system(f"cp {milvus_db_path} {db_path}")
        print(f"Copied milvus database to {db_path}")
    return milvus_dbs


def process_question(qa_pair, milvus_db_path):
    question = qa_pair['question']
    known_answer = qa_pair['answer']

    # Initialize the state
    state = DeRetSynState(
        original_question=question,
        model=os.getenv('TOGETHER_LLAMA31'),
        api_key=os.getenv('TOGETHER_API_KEY'),
        base_url="https://api.together.xyz/v1/",
        collection_name="surgical_information",
        verbose=False,
        milvus_directory=milvus_db_path
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

def run_evaluation(qa_dataset, num_processes, milvus_dbs):
    repeat_dbs = milvus_dbs * (len(qa_dataset) // len(milvus_dbs))
    repeat_dbs += milvus_dbs[:len(qa_dataset) % len(milvus_dbs)]
    combined_input = list(zip(qa_dataset, repeat_dbs))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_question, combined_input), total=len(qa_dataset)))

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
    num_processes = max(1, int(multiprocessing.cpu_count()/2.0))  # Use half of all available CPU cores

    # Create copies of the Milvus database
    milvus_dbs = milvus_copy_per_process(num_processes)

    print(f"Starting evaluation with {num_processes} processes...")
    results = run_evaluation(qa_dataset, num_processes, milvus_dbs)

    print_results(results)

    # Save the evaluation results to a file
    with open('surgical_qa_dataset_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Clean up the Milvus databases
    for db_path in milvus_dbs:
        os.remove(db_path)
        print(f"Deleted Milvus database: {db_path}")