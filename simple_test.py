from utils.agents import orchestrator, DeRetSynState
from dotenv import load_dotenv
import os

load_dotenv()

def simple_test(question: str="How should I remove an object stuck in someone's eye?"):
    state = DeRetSynState(
        original_question=question,
        model=os.getenv('TOGETHER_QWEN25_1dot5B'),
        api_key=os.getenv('TOGETHER_API_KEY'),
        base_url=os.getenv('TOGETHER_URL'),
        faiss_index_path="surgical_faiss_index",
        verbose=True,
        iterations=0,
        wikipedia_results="",
        run_async=False,
    )

    for step in orchestrator(state):
        if step['step'] == 'final':
            final_state = step['state']
            break

    return final_state['final_answer']