from utils.agents import orchestrator_straight_run, DeRetSynState
from utils.index_w_faiss import FaissReader
from dotenv import load_dotenv
import os

load_dotenv()

faiss_reader = FaissReader("surgical_faiss_index")

MODEL_NAME = os.getenv('TOGETHER_LLAMA32')
API_KEY = os.getenv('TOGETHER_API_KEY')
BASE_URL = "https://api.together.xyz/v1/"

state = DeRetSynState(
    original_question="What is the most common cause of acute appendicitis?",
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    faiss_index_path="surgical_faiss_index",
    verbose=True,
    iterations=0,
    wikipedia_results="",
    run_async=True,
    vectorstore=faiss_reader,
    fixed_context=None,
    retrieval_k=5,
)

final_state = orchestrator_straight_run(state)
print(final_state)
