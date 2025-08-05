import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import json
import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import logging
import asyncio
import multiprocessing
from functools import partial
from tqdm import tqdm
from utils.agents import orchestrator, DeRetSynState, evaluate_answer
from utils.llms import init_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def evaluate_pubmedqa_answer(generated_answer, known_answer):
    """Simple string matching evaluation for yes/no/maybe answers"""
    generated_clean = generated_answer.lower().strip()
    known_clean = known_answer.lower().strip()
    
    is_correct = generated_clean == known_clean
    
    evaluation = f"Generated: '{generated_clean}', Expected: '{known_clean}', Match: {is_correct}"
    
    return is_correct, evaluation

def process_question_wrapper(args):
    """Wrapper function for multiprocessing - unpacks arguments"""
    item, llm_name, eval_llm_name, use_implicit_knowledge, use_fixed_context, use_wikipedia_fallback = args
    return process_question(
        item, 
        llm_name,
        eval_llm_name=eval_llm_name,
        use_implicit_knowledge=use_implicit_knowledge,
        use_fixed_context=use_fixed_context,
        use_wikipedia_fallback=use_wikipedia_fallback
    )

def process_question(item, llm_name, eval_llm_name=None, use_implicit_knowledge=False, use_fixed_context=False, use_wikipedia_fallback=False):
    """Process a single question - modified to not write to file directly for parallel processing"""
    question = item['question']
    context = "\n".join(item['context']["contexts"])
    known_answer = item['final_decision']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Initialize evaluation LLM for this process
    eval_llm = init_llm(eval_llm_name) if eval_llm_name else None
    
    # Initialize the state - get_llm_object will handle the LLM configuration
    state = DeRetSynState(
        original_question=question,
        model=llm_name,
        faiss_index_path="surgical_faiss_index",
        verbose=True,
        api_key=None,
        base_url=None,
        iterations=0,
        wikipedia_results="",
        run_async=True,
        use_implicit_knowledge=use_implicit_knowledge,
        fixed_context=context if use_fixed_context else None,
        use_wikipedia_fallback=use_wikipedia_fallback,
        answer_choices=['yes','no','maybe']
    )
    
    try:
        # Run the synchronous orchestrator
        for step in orchestrator(state):
            if step['step'] == 'final':
                final_state = step['state']
                break
        
        # Extract the final answer and try to map it to yes/no/maybe
        final_answer = final_state['final_answer'].lower().strip()
        
        # Try to extract yes/no/maybe from the answer
        if 'yes' in final_answer and 'no' not in final_answer:
            generated_answer = 'yes'
        elif 'no' in final_answer and 'yes' not in final_answer:
            generated_answer = 'no'
        elif 'maybe' in final_answer or 'uncertain' in final_answer or 'unclear' in final_answer:
            generated_answer = 'maybe'
        else:
            # Default fallback - try to determine from context
            generated_answer = 'maybe'
        
        # Use evaluate_answer function for evaluation
        is_correct = evaluate_answer(final_state, known_answer, eval_llm)
        
        # Also keep the simple string matching for comparison
        simple_is_correct, simple_evaluation = evaluate_pubmedqa_answer(generated_answer, known_answer)
        
        result = {
            'question': question,
            'context': context if use_fixed_context else None,
            'document_context': final_state.get('answers', ''),
            'wikipedia_context': final_state.get('wikipedia_results', ''),
            'cot': final_state.get('cot_for_answer', ''),
            'full_rag_answer': final_state['final_answer'],
            'extracted_answer': generated_answer,
            'known_answer': known_answer,
            'is_correct': is_correct,
            'simple_is_correct': simple_is_correct,
            'simple_evaluation': simple_evaluation,
            'used_implicit_knowledge': use_implicit_knowledge,
            'used_fixed_context': use_fixed_context,
            'iterations': final_state['iterations']
        }
        
        logging.info(f"Generated answer: {generated_answer}, Known answer: {known_answer}, LLM Eval: {is_correct}, Simple Eval: {simple_is_correct}")
        
    except Exception as e:
        logging.error(f"Error running orchestrator for question {question[:50]}...: {str(e)}")
        result = {
            'question': question,
            'context': context if use_fixed_context else None,
            'document_context': None,
            'wikipedia_context': None,
            'cot': None,
            'full_rag_answer': None,
            'extracted_answer': 'maybe',
            'known_answer': known_answer,
            'is_correct': False,
            'simple_is_correct': False,
            'simple_evaluation': f"Error: {str(e)}",
            'used_implicit_knowledge': use_implicit_knowledge,
            'used_fixed_context': use_fixed_context,
            'iterations': 0,
            'verbose': True,
        }

    logging.info(f"Question processed: {question[:50]}...")
    return result

async def process_question_async(item, llm_name, eval_llm_name=None, use_implicit_knowledge=False, use_fixed_context=False, use_wikipedia_fallback=False):
    """Async version of process_question"""
    question = item['question']
    context = "\n".join(item['context']["contexts"])
    known_answer = item['final_decision']
    
    logging.info(f"Processing question: {question[:50]}...")
    
    # Initialize evaluation LLM
    eval_llm = init_llm(eval_llm_name) if eval_llm_name else None
    
    # Initialize the state
    state = DeRetSynState(
        original_question=question,
        model=llm_name,
        faiss_index_path="surgical_faiss_index",
        verbose=True,
        api_key=None,
        base_url=None,
        iterations=0,
        wikipedia_results="",
        run_async=True,
        use_implicit_knowledge=use_implicit_knowledge,
        fixed_context=context if use_fixed_context else None,
        use_wikipedia_fallback=use_wikipedia_fallback,
        answer_choices=['yes','no','maybe']
    )
    
    try:
        # Run the orchestrator - use regular for loop, not async for
        for step in orchestrator(state):
            if step['step'] == 'final':
                final_state = step['state']
                break
        
        # Extract the final answer and try to map it to yes/no/maybe
        final_answer = final_state['final_answer'].lower().strip()
        
        # Try to extract yes/no/maybe from the answer
        if 'yes' in final_answer and 'no' not in final_answer:
            generated_answer = 'yes'
        elif 'no' in final_answer and 'yes' not in final_answer:
            generated_answer = 'no'
        elif 'maybe' in final_answer or 'uncertain' in final_answer or 'unclear' in final_answer:
            generated_answer = 'maybe'
        else:
            # Default fallback - try to determine from context
            generated_answer = 'maybe'
        
        # Use evaluate_answer function for evaluation
        is_correct = evaluate_answer(final_state, known_answer, eval_llm)
        
        # Also keep the simple string matching for comparison
        simple_is_correct, simple_evaluation = evaluate_pubmedqa_answer(generated_answer, known_answer)
        
        result = {
            'question': question,
            'context': context if use_fixed_context else None,
            'document_context': final_state.get('answers', ''),
            'wikipedia_context': final_state.get('wikipedia_results', ''),
            'cot': final_state.get('cot_for_answer', ''),
            'full_rag_answer': final_state['final_answer'],
            'extracted_answer': generated_answer,
            'known_answer': known_answer,
            'is_correct': is_correct,
            'simple_is_correct': simple_is_correct,
            'simple_evaluation': simple_evaluation,
            'used_implicit_knowledge': use_implicit_knowledge,
            'used_fixed_context': use_fixed_context,
            'iterations': final_state['iterations']
        }
        
        logging.info(f"Generated answer: {generated_answer}, Known answer: {known_answer}, LLM Eval: {is_correct}, Simple Eval: {simple_is_correct}")
        
    except Exception as e:
        logging.error(f"Error running orchestrator for question {question[:50]}...: {str(e)}")
        result = {
            'question': question,
            'context': context if use_fixed_context else None,
            'document_context': None,
            'wikipedia_context': None,
            'cot': None,
            'full_rag_answer': None,
            'extracted_answer': 'maybe',
            'known_answer': known_answer,
            'is_correct': False,
            'simple_is_correct': False,
            'simple_evaluation': f"Error: {str(e)}",
            'used_implicit_knowledge': use_implicit_knowledge,
            'used_fixed_context': use_fixed_context,
            'iterations': 0,
            'verbose': True,
        }

    logging.info(f"Question processed: {question[:50]}...")
    return result

def append_to_json_file(result, results_file):
    """Append a single result to the JSON file"""
    try:
        with open(results_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    except Exception as e:
        logging.error(f"Error appending to file {results_file}: {str(e)}")

def run_evaluation_multiprocessing(dataset_list, num_processes, llm_name, eval_llm_name, use_implicit_knowledge, use_fixed_context, use_wikipedia_fallback, results_file):
    """Run evaluation using multiprocessing"""
    
    # Prepare arguments for each item
    args_list = [
        (item, llm_name, eval_llm_name, use_implicit_knowledge, use_fixed_context, use_wikipedia_fallback) for item in dataset_list
    ]
    
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_question_wrapper, args_list), total=len(dataset_list)):
                if result:
                    append_to_json_file(result, results_file)
                    results.append(result)
    else:
        results = []
        for args in tqdm(args_list, total=len(dataset_list)):
            result = process_question_wrapper(args)
            if result:
                append_to_json_file(result, results_file)
                results.append(result)

    return results

async def run_evaluation_async(dataset_list, llm_name, eval_llm_name, use_implicit_knowledge, use_fixed_context, use_wikipedia_fallback, results_file):
    """Run evaluation using async programming"""
    
    async def process_and_save(item):
        result = await process_question_async(
            item, 
            llm_name,
            eval_llm_name=eval_llm_name,
            use_implicit_knowledge=use_implicit_knowledge,
            use_fixed_context=use_fixed_context,
            use_wikipedia_fallback=use_wikipedia_fallback
        )
        append_to_json_file(result, results_file)
        return result
    
    # Create tasks for all questions
    tasks = [process_and_save(item) for item in dataset_list]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system on PubMedQA dataset using DeRetSynState and orchestrator')
    parser.add_argument('--llm', type=str, default='azure-gpt4', 
                       help='LLM model to use (e.g., azure-gpt4, azure-gpt35, together-llama33)')
    parser.add_argument('--eval_llm', type=str, default=None,
                       help='LLM model to use for evaluation (default: same as --llm)')
    parser.add_argument('--num_questions', type=int, default=None,
                       help='Number of questions to evaluate (default: all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file name (default: auto-generated based on model)')
    parser.add_argument('--use_implicit_knowledge', action='store_true',
                       help='Use LLM implicit knowledge instead of document retrieval')
    parser.add_argument('--use_fixed_context', action='store_true',
                       help='Use the context from PubMedQA dataset as fixed context')
    parser.add_argument('--use_wikipedia_fallback', action='store_true',
                       help='Enable Wikipedia fallback when document retrieval is insufficient')
    parser.add_argument('--use_async', action='store_true',
                       help='Run evaluation asynchronously instead of using multiprocessing')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of processes to use for multiprocessing (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.use_implicit_knowledge and args.use_fixed_context:
        parser.error("Cannot use both --use_implicit_knowledge and --use_fixed_context at the same time")
    
    logging.info("Starting the PubMedQA RAG evaluation process...")
    logging.info(f"Using implicit knowledge: {args.use_implicit_knowledge}")
    logging.info(f"Using fixed context: {args.use_fixed_context}")
    logging.info(f"Using Wikipedia fallback: {args.use_wikipedia_fallback}")
    logging.info(f"Running asynchronously: {args.use_async}")
    
    logging.info(f"Using model: {args.llm}")
    
    # Initialize evaluation LLM name
    eval_llm_name = args.eval_llm if args.eval_llm else args.llm
    logging.info(f"Using evaluation model: {eval_llm_name}")
    
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
        eval_model_name = eval_llm_name.replace('-', '_')
        if args.use_implicit_knowledge:
            mode_suffix = 'implicit_knowledge'
        elif args.use_fixed_context:
            mode_suffix = 'fixed_context'
        elif args.use_wikipedia_fallback:
            mode_suffix = 'wikipedia_fallback'
        else:
            mode_suffix = 'rag_retrieval'
        results_file = f'pubmedqa_deretsyn_results_{model_name}_{mode_suffix}_eval_{eval_model_name}.json'
    
    # Initialize the results file
    with open(results_file, 'w') as f:
        json.dump([], f)

    logging.info(f"Processing {len(dataset_list)} questions...")

    # Determine number of processes
    if args.num_processes:
        num_processes = args.num_processes
    else:
        num_processes = multiprocessing.cpu_count()

    # Run evaluation
    if args.use_async:
        results = asyncio.run(run_evaluation_async(dataset_list, args.llm, eval_llm_name, args.use_implicit_knowledge, args.use_fixed_context, args.use_wikipedia_fallback, results_file))
    else:
        results = run_evaluation_multiprocessing(dataset_list, num_processes, args.llm, eval_llm_name, args.use_implicit_knowledge, args.use_fixed_context, args.use_wikipedia_fallback, results_file)

    # Calculate accuracy for both evaluation methods
    llm_accuracy = sum(1 for result in results if result['is_correct']) / len(results)
    simple_accuracy = sum(1 for result in results if result['simple_is_correct']) / len(results)
    
    # Calculate per-class accuracy for LLM evaluation
    yes_correct = sum(1 for result in results if result['known_answer'].lower() == 'yes' and result['is_correct'])
    no_correct = sum(1 for result in results if result['known_answer'].lower() == 'no' and result['is_correct'])
    maybe_correct = sum(1 for result in results if result['known_answer'].lower() == 'maybe' and result['is_correct'])
    
    yes_total = sum(1 for result in results if result['known_answer'].lower() == 'yes')
    no_total = sum(1 for result in results if result['known_answer'].lower() == 'no')
    maybe_total = sum(1 for result in results if result['known_answer'].lower() == 'maybe')

    # Calculate average iterations
    avg_iterations = sum(result['iterations'] for result in results) / len(results)

    logging.info(f"Evaluation completed.")
    logging.info(f"LLM Evaluation Accuracy: {llm_accuracy:.2%}")
    logging.info(f"Simple String Matching Accuracy: {simple_accuracy:.2%}")
    logging.info(f"Average iterations: {avg_iterations:.2f}")
    if yes_total > 0:
        logging.info(f"'Yes' Accuracy (LLM Eval): {yes_correct/yes_total:.2%} ({yes_correct}/{yes_total})")
    if no_total > 0:
        logging.info(f"'No' Accuracy (LLM Eval): {no_correct/no_total:.2%} ({no_correct}/{no_total})")
    if maybe_total > 0:
        logging.info(f"'Maybe' Accuracy (LLM Eval): {maybe_correct/maybe_total:.2%} ({maybe_correct}/{maybe_total})")
    
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()