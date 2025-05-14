import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import multiprocessing
from tqdm import tqdm
import sys
import time
import logging
from utils.index_w_faiss import FaissReader
from utils.agents import evaluate_answer
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Rate limiting constants
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_PERIOD = 60  # seconds

def load_qa_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_eval_results(file_path: str="react_rag_evaluation_results_llama32.json"):
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
react_llm = ChatOpenAI(
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

# Create a document search tool using FaissReader
def search_documents(query: str) -> str:
    """Search for relevant surgical information in the document database."""
    try:
        results = faiss_reader.search(query, k=5)
        return results
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Create the tool
document_search_tool = Tool(
    name="DocumentSearch",
    func=search_documents,
    description="Useful for searching surgical information in the medical document database. Input should be a search query related to surgical procedures, conditions, or techniques."
)

# Create the ReAct agent
system_prompt = """You are a medical assistant specializing in surgical information. Your goal is to provide accurate and helpful information about surgical procedures, techniques, and related medical knowledge.

When answering questions:
1. Use the DocumentSearch tool to find relevant information in the surgical database
2. Analyze the retrieved information carefully
3. Provide comprehensive and accurate answers based on the retrieved content
4. If the information is not available in the search results, acknowledge the limitations
5. Think step-by-step to reason through complex questions
6. Cite specific parts of the retrieved documents when appropriate
7. Focus on providing factual medical information rather than opinions

Remember accuracy is crucial. Provide all reasoning and the final answer.
"""

class SurgInfoResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    reasoning: str = Field(description="The reasoning process used to arrive at the final answer.")

react_agent = create_react_agent(
    model=react_llm,
    tools=[document_search_tool],
    prompt=system_prompt,
    response_format=SurgInfoResponse,
)

def append_to_json_file(result: dict, file_path: str="react_rag_evaluation_results_llama32.json"):
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

def extract_final_answer(agent_response):
    """Extract the final answer from the agent's response."""
    # The final answer is in the last agent output
    return agent_response["messages"][-1].content

def extract_reasoning(agent_response):
    """Extract the reasoning process from the agent's response."""
    # Try to extract the intermediate steps if available
    if hasattr(agent_response, "intermediate_steps") and agent_response.intermediate_steps:
        reasoning = []
        for step in agent_response.intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step
                reasoning.append(f"Action: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}")
        return "\n\n".join(reasoning)
    
    # If we can't extract intermediate steps, return a placeholder
    return "Reasoning process not available in this format"

def process_question(qa_pair):
    question = qa_pair['question']
    if question in ALL_EVAL_QUESTIONS_SO_FAR:
        logging.info(f"Skipping evaluation for question {question} (already evaluated)")
        return None
    
    known_answer = qa_pair['answer']

    try:
        # Run the ReAct agent
        messages = {"messages": [{"role": "user", "content": question}]}
        agent_response = react_agent.invoke(messages)
        
        
        # Extract the final answer and reasoning
        final_answer = extract_final_answer(agent_response)
        reasoning = extract_reasoning(agent_response)
        
        # Extract context from the reasoning if possible
        context = ""
        for step in agent_response.intermediate_steps:
            if step[0].tool == "DocumentSearch":
                context += step[1] + "\n\n"
        
        # Create a state-like object for evaluation
        state = {
            "original_question": question,
            "final_answer": final_answer,
            "cot_for_answer": reasoning,
            "verbose": False
        }
        
        # Evaluate the answer
        is_correct = evaluate_answer(state, known_answer, eval_llm)

        output = {
            'question': question,
            'document_context': context.strip(),
            'cot': reasoning,
            'rag_answer': final_answer,
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

def run_evaluation(qa_dataset, num_processes, num_questions=None):
    if num_questions:
        qa_dataset = qa_dataset[:num_questions]
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
    # grab num_processes from command line arguments
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # limit to a number of questions for testing
    num_questions = int(sys.argv[2]) if len(sys.argv) > 2 else None  # default to 100 questions

    # Load the QA dataset
    qa_dataset = load_qa_dataset('surgical_qa_dataset.json')

    # Set the number of processes to use
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count()/1.5))  # Use half of all available CPU cores
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count() - 1)  # Limit to the number of available CPU cores

    print(f"Starting evaluation with {num_processes} processes...")
    results = run_evaluation(qa_dataset, num_processes, num_questions)

    print_results(results)

    # Save the evaluation results to a file
    with open('react_rag_evaluation_results_llama32.json', 'w') as f:
        json.dump(results, f, indent=4)