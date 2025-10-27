import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from typing import List
from utils.index_w_faiss import FaissReader
import dspy
from typing import TypedDict, List
from utils.wikipedia_helps import grab_wikipedia_context
import asyncio
import functools
from utils.llms import init_llm

def to_thread(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    return wrapper

# Define the structure of your state
class DeRetSynState(TypedDict):
    original_question: str=""
    answers: str=""
    iterations: int=0
    model: str=""
    api_key: str=None
    base_url: str=None
    done: bool=False
    wikipedia_results: str=None
    pending_queries: List[str]=[]
    final_answer: str=None
    final_confidence: str=None
    cot_for_answer: str=None
    verbose: bool=False
    faiss_index_path: str="surgical_faiss_index"
    retrieval_k: int=5
    run_async: bool=False
    use_implicit_knowledge: bool=False
    fixed_context: str=None  # New field for user-provided fixed context
    use_wikipedia_fallback: bool=True  # New field to control Wikipedia fallback
    answer_choices: List[str]=None
    vectorstore: FaissReader=None

decomposition_prompt = PromptTemplate.from_template(
    """You are an expert at breaking complex surgical questions into simpler ones. Break the following question into smaller sub-questions:
    
## Question
{question}
----

Each sub-question should be independent and answerable on it's own without needing reference to other sub-questions. Think of at least 3 sub-questions but no more than 7.
    
Think step-by-step and make sure to reason through how break the question into sub-questions. 
    
Create new sub-questions in the following format but DO NOT answer the question. Respond in the following format:
    
<think> Your reasoning here... </think>
<sub-question> The first sub-question... </sub-question>
<sub-question> The second sub-question... </sub-question>
...
<sub-question> The last sub-question... </sub-question>"""
)

def get_llm_object(state: DeRetSynState):
    # Check if we can use the standard init_llm approach
    if state.get('api_key') and state.get('base_url'):
        # Use direct initialization for dynamic API keys/URLs
        if "ollama" not in state['api_key']:
            return ChatOpenAI(model=state["model"],
                         api_key=state["api_key"],
                         base_url=state["base_url"],
                         temperature=0.7)
        else:
            return ChatOllama(model=state["model"],
                              api_key=state["api_key"],
                              base_url=state["base_url"],
                              num_ctx=32000,
                              temperature=0.7)
    else:
        # Fall back to init_llm for environment-based configuration
        return init_llm(state["model"], llm_temperature=0.7)

def agent_a_decompose_question(state: DeRetSynState) -> bool:
    llm = get_llm_object(state)
    original_question = state["original_question"]
    prompt = decomposition_prompt.format(question=original_question)
    try:
        full_response = llm.invoke(prompt).content.strip()
        sub_questions = full_response.strip().split("<sub-question>")[1:]
        if len(sub_questions) < 2:
            logging.error(f"Not enough sub-questions generated for question: {original_question}")
            return False
        sub_questions = [sub_q.split("</sub-question>")[0].strip() for sub_q in sub_questions]
        for sub_q in sub_questions:
            if len(sub_q) < 2:
                logging.error(f"Sub-question too short: {sub_q}")
                return False
        if state["verbose"]:
            print(f"Full response for sub-question generation: {full_response}")
            print(f"Initial sub-questions: {sub_questions}")
        state["pending_queries"] = sub_questions
        return True
    except Exception as e:
        logging.error(f"Error decomposing question: {str(e)}")
        return False


def generate_answer_from_implicit_knowledge(state: DeRetSynState, question: str) -> str:
    """Generate answer using LLM's implicit knowledge without vectorstore search"""
    llm = get_llm_object(state)
    prompt = f"""You are a medical expert specializing in surgery. Answer the following question using your knowledge of surgical procedures, anatomy, and medical practices.

## Question
{question}
----

Think step-by-step and provide a comprehensive answer based on your medical knowledge. If you're uncertain about any aspect, please indicate that in your response.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer based on your medical knowledge... </answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>"""
    
    full_response = llm.invoke(prompt).content.strip()
    if state["verbose"]:
        print(f"Generated answer from implicit knowledge: {full_response}")
    
    try:
        response = full_response.split("<answer>")[1].split("</answer>")[0].strip()
        confidence = full_response.split("<confidence>")[1].split("</confidence>")[0].strip()
        return response, f"Confidence: {confidence}"
    except IndexError:
        # Fallback if the format is not followed
        return full_response, "Confidence: Unknown"


async def generate_answer_from_implicit_knowledge_async(state: DeRetSynState, question: str) -> str:
    """Async version of generate_answer_from_implicit_knowledge"""
    llm = get_llm_object(state)
    prompt = f"""You are a medical expert specializing in surgery. Answer the following question using your knowledge of surgical procedures, anatomy, and medical practices.

## Question
{question}
----

Think step-by-step and provide a comprehensive answer based on your medical knowledge. If you're uncertain about any aspect, please indicate that in your response.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer based on your medical knowledge... </answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>"""
    
    response = await to_thread(llm.invoke)(prompt)
    content = response.content.strip()
    if state["verbose"]:
        print(f"Generated answer from implicit knowledge (async): {content}")
    
    try:
        answer = content.split("<answer>")[1].split("</answer>")[0].strip()
        confidence = content.split("<confidence>")[1].split("</confidence>")[0].strip()
        return answer, f"Confidence: {confidence}"
    except IndexError:
        # Fallback if the format is not followed
        return content, "Confidence: Unknown"


def agent_b_retrieve(state: DeRetSynState) -> None:
    queries = state["pending_queries"]
    answers = state.get("answers", "")
    new_answers = []

    if state.get("use_implicit_knowledge", False):
        # Use LLM's implicit knowledge
        for q in queries:
            response, confidence = generate_answer_from_implicit_knowledge(state, q)
            answer_text = f"Question: {q}\nAnswer: {response}\n{confidence}\n\n\n"
            new_answers.append(answer_text)
    else:
        for q in queries:  # TODO: make these calls asynchronously
            if state["fixed_context"]:
                results = state["fixed_context"]
            else:
                # Use vectorstore search (original behavior)
                vectorstore = state['vectorstore']
                results = vectorstore.search(q, k=state["retrieval_k"])
            response = None
            while response is None:
                try:
                    response, confidence, snippets = generate_answer_from_question_and_context(state, q, results)
                except Exception as e:
                    logging.error(f"Error generating answer from question and context: {str(e)}")
                    response = None
            answer_text = f"Question: {q}\nAnswer: {response}\nConfidence: {confidence}\n\n\n"
            new_answers.append(answer_text)
    
    combined_answers = "".join(new_answers)
    if state["verbose"]:
        print(f"New answers: {combined_answers}")
    state["answers"] = answers + combined_answers
    state["pending_queries"] = []


async def agent_b_retrieve_async(state: DeRetSynState) -> None:
    queries = state["pending_queries"]
    answers = state.get("answers", "")

    async def process_query(q):
        if state.get("use_implicit_knowledge", False):
            # Use LLM's implicit knowledge
            response, confidence = await generate_answer_from_implicit_knowledge_async(state, q)
            return f"Question: {q}\nAnswer: {response}\n{confidence}\n\n\n"
        else:
            # Use vectorstore search (original behavior)
            if state["fixed_context"]:
                results = state["fixed_context"]
            else:
                vectorstore = state['vectorstore']
                results = await to_thread(vectorstore.search)(q, k=state["retrieval_k"])
            response, confidence, snippets = await generate_answer_from_question_and_context_async(state, q, results)
            return f"Question: {q}\nAnswer: {response}\nConfidence: {confidence}\n\n\n"
    
    # Run all queries concurrently
    tasks = [process_query(q) for q in queries]
    new_answers = await asyncio.gather(*tasks)

    combined_answers = "".join(new_answers)
    if state["verbose"]:
        print(f"New answers: {combined_answers}")
    state["answers"] = answers + combined_answers
    state["pending_queries"] = []

def generate_answer_from_question_and_context(state: DeRetSynState,
                                              question: str,
                                              context: str) -> str:
    llm = get_llm_object(state)
    prompt = f"""Based on the given question and context, generate an answer.
## Question
{question}
----
## Context
{context}
----

Think step-by-step and make sure to reason through how to generate an answer. Rely on the given context and your own knowledge to generate the answer. 

Include snippets of the context that support your answer.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer... </answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>
<snippet> First relevant snippet from the context... </snippet>
<snippet> Second relevant snippet from the context... </snippet>
...
<snippet> The last relevant snippet from the context </snippet>"""
    try:
        full_response = llm.invoke(prompt).content.strip()
        if state["verbose"]:
            print(f"Generated answer for question and context: {full_response}")
        if "<answer>" not in full_response or "</answer>" not in full_response or "<confidence>" not in full_response or "</confidence>" not in full_response or "<snippet>" not in full_response or "</snippet>" not in full_response:
            return None, None, None
        response = full_response.split("<answer>")[1].split("</answer>")[0].strip()
        if len(response) == 0:
            return None, None, None
        confidence = full_response.split("<confidence>")[1].split("</confidence>")[0].strip()
        if len(confidence) == 0:
            return None, None, None
        snippets = full_response.split("<snippet>")[1:-1]
        snippets = [snippet.split("</snippet>")[0].strip() for snippet in snippets]
        if len(snippets) == 0:
            return None, None, None
        for snippet in snippets:
            if len(snippet) == 0:
                return None, None, None
        if state["verbose"]:
            print(f"Full response for generating answer from question and context: {full_response}")
        return response, confidence, "\n".join(snippets)
    except Exception as e:
        logging.error(f"Error generating answer from question and context: {str(e)}")
        return None, None, None


async def generate_answer_from_question_and_context_async(state: DeRetSynState,
                                                          question: str,
                                                          context: str) -> str:
    llm = get_llm_object(state)
    prompt = f"""Based on the given question and context, generate an answer.
## Question
{question}
----
## Context
{context}
----

Think step-by-step and make sure to reason through how to generate an answer. Rely on the given context and your own knowledge to generate the answer. 

Include snippets of the context that support your answer.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer... </answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>
<snippet> First relevant snippet from the context... </snippet>
<snippet> Second relevant snippet from the context... </snippet>
...
<snippet> The last relevant snippet from the context </snippet>"""
    response = await to_thread(llm.invoke)(prompt)
    content = response.content.strip()
    answer = content.split("<answer>")[1].split("</answer>")[0].strip()
    confidence = content.split("<confidence>")[1].split("</confidence>")[0].strip()
    snippets = content.split("<snippet>")[1:-1]
    snippets = [snippet.split("</snippet>")[0].strip() for snippet in snippets]
    if state["verbose"]:
        print(f"Generated answer for question and context (async): {content}")
    return answer, confidence, "\n".join(snippets)


def agent_c_synthesize(state: DeRetSynState) -> bool:
    llm = get_llm_object(state)
    original_question = state["original_question"]
    answers = state["answers"]
    answer_choices = state.get("answer_choices", None)
    fixed_context = state.get("fixed_context", None)
    if fixed_context:
        context_string = f"{fixed_context}\n"
    else:
        context_string = ""

    check_prompt = f"""
You are a reasoning engine. Given the following sub-question answers, determine whether they are enough to fully answer the original question. Rely on the provided context to determine whether the question can be answered.

If yes, then provide the answer. Make your answer detailed and structured with sections, as appropriate. Include as much relevant information as possible from the knowledge provided.

If you determine that you cannot answer the original question, then suggest what additional questions should be asked.

## Original Question
{original_question}
----

## Knowledge
{context_string}{answers}
----
"""
    middle_prompt = ""
    answer_string = " The answer to the original question... "
    if answer_choices:
        answer_string = " / ".join(answer_choices)
        choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(answer_choices)])
        middle_prompt = f"""This is a multiple choice question. You must select one of the following options:
{choices_text}

"""

    suffix = f"""Think step-by-step to reason through you answer and consider the relevant information. Respond in the following format:
<think> Your reasoning here... </think>
<can_answer> yes OR no </can_answer>
<answer>{answer_string}</answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>
<new_questions> The first new sub-question... </new_questions>
<new_questions> The second new sub-question... </new_questions>
...
<new questions> The last new sub-question </new_questions>
"""
    try:
        response = llm.invoke(check_prompt+middle_prompt+suffix).content.strip()
        if state["verbose"]:
            print(f"Synthesizer response: {response}")

        if "<think>" not in response or "</think>" not in response or "<can_answer>" not in response or "</can_answer>" not in response:
            return False
        can_answer = response.split("<can_answer>")[1].split("</can_answer>")[0].strip().lower()
        if can_answer == "yes":
            if "<answer>" not in response or "</answer>" not in response:
                return False
            answer_text = response.split("<answer>")[1].split("</answer>")[0].strip()
            if "<confidence>" not in response or "</confidence>" not in response:
                return False
            confidence = response.split("<confidence>")[1].split("</confidence>")[0].strip()
            state["done"] = True
            state["final_answer"] = answer_text
            state["iterations"] = 1
            state["final_confidence"] = confidence
            return True
        else:
            state["done"] = False
            state["iterations"] += 1
        new_queries = []
        if "<new_questions>" not in response or "</new_questions>" not in response:
            return False
        new_q_block = response.split("<new_questions>")[1:]
        new_queries = [q.split("</new_questions>")[0].strip() for q in new_q_block]
        if len(new_queries) == 0:
            return False
        state["pending_queries"] = new_queries
        return True
    except Exception as e:
        logging.error(f"Error running synthesizer for question {state['original_question']}: {str(e)}")
        return False


def agent_d_best_effort(state: DeRetSynState):
    original_question = state["original_question"]
    answer_choices = state.get("answer_choices", None)
    middle_prompt = ""
    answer_string = "The answer to the original question..."
    if answer_choices:
        answer_string = " / ".join(answer_choices)
        choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(answer_choices)])
        middle_prompt = f"""

This is a multiple choice question. You must select one of the following options:
{choices_text}

"""

    search_wikipedia(state)
    generate_prompt = f"""
You are a reasoning engine. Given the following original question and sub-question answers, formulate an answer to the best of your ability.

## Original Question
{original_question}
----

## Context
{state["answers"]}
{state["wikipedia_results"]}
{middle_prompt}
----

Think step-by-step to reason through you answer and consider the provided context as well as your own knowledge.

Respond in the following format:
<think> Your reasoning here... </think>
<answer> {answer_string} </answer>
<confidence> High/Medium/Low - your confidence level in this answer </confidence>
"""
    try:
        llm = get_llm_object(state)
        response = llm.invoke(generate_prompt).content.strip()
        if state["verbose"]:
            print(f"Best-effort response with help from Wikipedia: {response}")
        if "<answer>" not in response or "</answer>" not in response:
            return False
        answer_text = response.split("<answer>")[1].split("</answer>")[0].strip()
        answer_text += "\n\n" + "NOTE: I could not answer the question completely with the available documents. I have tried to use Wikipedia to help me answer the question to the best of my ability."
        if "<confidence>" not in response or "</confidence>" not in response:
            return False
        confidence = response.split("<confidence>")[1].split("</confidence>")[0].strip()
        state["done"] = True
        state["final_answer"] = answer_text
        state["final_confidence"] = confidence
        return True
    except Exception as e:
        logging.error(f"Error running best-effort agent for question {state['original_question']}: {str(e)}")
        return False

def search_wikipedia(state: DeRetSynState) -> str:
    results_fast = search_wikipedia_fast(state["original_question"])
    results_slow = search_wikipedia_slow(state["original_question"])
    state["wikipedia_results"] = results_fast + "\n\n" + results_slow

def search_wikipedia_fast(query: str) -> str:
    try:
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=state["retrieval_k"])
        new_answer = "\n\n".join([x['text'] for x in results])
        return new_answer
    except Exception as e:
        return "Could not retrieve information from Wikipedia from a fast search."

def search_wikipedia_slow(query: str) -> str:
    try:
        results = grab_wikipedia_context(query)
        return results
    except Exception as e:
        return "Could not retrieve information from Wikipedia from a slow search."


def agent_e_follow_up_question_generator(state: DeRetSynState) -> bool:
    original_question = state["original_question"]
    final_answer = state["final_answer"]
    prompt = f"""You are a reasoning engine. Given the following original question and final answer, generate 3 follow-up questions that help expand on the original question and the answer in a step-wise manner.

## Original Question:
{original_question}
----

## Final Answer:
{final_answer}

Think step-by-step to reason through your answer and consider the relevant information. Respond in the following format:
<think> Your reasoning here... </think>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>"""
    try:
        llm = get_llm_object(state)
        response = llm.invoke(prompt).content.strip()
        if state["verbose"]:
            print(f"Follow-up questions response: {response}")
        if "<follow_up_questions>" not in response or "</follow_up_questions>" not in response:
            return False
        follow_up_questions = response.split("<follow_up_questions>")[1:-1]
        follow_up_questions = [q.split("</follow_up_questions>")[0].strip() for q in follow_up_questions]
        if len(follow_up_questions) < 1:
            return False
        state["pending_queries"] = follow_up_questions
        return True
    except Exception as e:
        logging.error(f"Error running follow-up question generator for question {state['original_question']}: {str(e)}")
        return False


def agent_f_cot_generator(state: DeRetSynState) -> bool:
    prompt = f"""
You are a reasoning engine. Based on the following question and knowledge, provide a detailed, step-by-step reasoning to arrive at an answer. Include at least 3 steps, but more as needed.

## Question:
{state["original_question"]}
----

## Knowledge:
{state["answers"]}
{state["wikipedia_results"] if "wikipedia_results" in state else ""}
----

Provide your response in this format:

<think> Your reasoning here... </think>
<answer> The final answer here... </answer>
"""
    llm = get_llm_object(state)
    response = llm.invoke(prompt).content.strip()
    if state["verbose"]:
        print(f"COT response: {response}")
    if "<think>" not in response or "</think>" not in response:
        return False
    cot = response.split("<think>")[1].split("</think>")[0].strip()
    state['cot_for_answer'] = cot
    return True

def orchestrator_straight_run(state: DeRetSynState):
    # Step 1: Decompose the question
    agent_a_complete = False
    while not agent_a_complete:
        agent_a_complete = agent_a_decompose_question(state)
    
    keep_going = True
    while keep_going:
        # Step 2: Retrieve relevant documents
        agent_b_retrieve(state)
        
        # Step 3: Synthesize the answer
        agent_c_complete = False
        while not agent_c_complete:
            agent_c_complete = agent_c_synthesize(state)
        
        # Check if we are done
        keep_going = not state["done"]

        # Stop after 3 iterations
        if state["iterations"] >= 2 and keep_going:
            if state["use_wikipedia_fallback"]:
                agend_d_complete = False
                while not agend_d_complete:
                    agend_d_complete = agent_d_best_effort(state)
                keep_going = False
    
    agent_e_complete = False
    while not agent_e_complete:
        agent_e_complete = agent_e_follow_up_question_generator(state)
    
    # generate COT
    agent_f_complete = False
    while not agent_f_complete:
        agent_f_complete = agent_f_cot_generator(state)
    
    # Return the final answer
    return state

async def orchestrator_async(state: DeRetSynState):
    # Step 1: Decompose the question
    agent_a_decompose_question(state)
    yield {"step": "decompose_complete", "sub_questions": state["pending_queries"]}

    keep_going = True
    while keep_going:
        # Step 2: Retrieve relevant documents
        await agent_b_retrieve_async(state)
        yield {"step": "retrieve_complete", "answers": state["answers"]}

        # Step 3: Synthesize the answer
        agent_c_synthesize(state)
        yield {"step": "synthesize_complete", "done": state["done"], "final_answer": state.get("final_answer"), "new_queries": state.get("pending_queries")}

        # Check if we are done
        keep_going = not state["done"]

        if state["iterations"] >= 2 and keep_going:
            if state["use_wikipedia_fallback"]:
                yield {"step": "start_best_effort"}
                # Step 4: Best effort answer
                agent_d_best_effort(state)
                agent_e_follow_up_question_generator(state)
                yield {"step": "best_effort_complete", "wiki_results": state["wikipedia_results"], "final_answer": state["final_answer"]}
                keep_going = False

    # generate COT
    agent_f_cot_generator(state)
    
    # Return the final answer
    yield {"step": "final", "state": state}

def orchestrator(state: DeRetSynState):
    # Step 1: Decompose the question
    agent_a_decompose_question(state)
    yield {"step": "decompose_complete", "sub_questions": state["pending_queries"]}

    keep_going = True
    while keep_going:
        # Step 2: Retrieve relevant documents
        if state["run_async"]:
            # Run async version in a new event loop
            import asyncio
            asyncio.run(agent_b_retrieve_async(state))
        else:
            # Run sync version
            agent_b_retrieve(state)
        
        yield {"step": "retrieve_complete", "answers": state["answers"]}

        # Step 3: Synthesize the answer
        agent_c_synthesize(state)
        yield {"step": "synthesize_complete", "done": state["done"], "final_answer": state.get("final_answer"), "new_queries": state.get("pending_queries")}

        # Check if we are done
        keep_going = not state["done"]

        if state["iterations"] >= 2 and keep_going:
            if state["use_wikipedia_fallback"]:
                yield {"step": "start_best_effort"}
                # Step 4: Best effort answer
                agent_d_best_effort(state)
                agent_e_follow_up_question_generator(state)
                yield {"step": "best_effort_complete", "wiki_results": state["wikipedia_results"], "final_answer": state["final_answer"]}
                keep_going = False

    # generate COT
    agent_f_cot_generator(state)
    
    # Return the final answer
    yield {"step": "final", "state": state}

def orchestrator_sync_wrapper(state: DeRetSynState):
    """Wrapper to handle async orchestrator in sync context"""
    async def run_async_orchestrator():
        results = []
        async for step in orchestrator_async(state):
            results.append(step)
        return results
    
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, so we need to use a different approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_async_orchestrator())
            results = future.result()
    except RuntimeError:
        # No running loop, we can use asyncio.run
        results = asyncio.run(run_async_orchestrator())
    
    # Yield the results
    for result in results:
        yield result

def evaluate_answer(state: DeRetSynState, known_answer: str, llm: ChatOpenAI=None) -> bool:
    prompt = f"""You are a medical reasoning engine that compares a ground truth answer against a generated answer to a given question to determine whether the generated answer is correct. Here is the question and the two answers:

Question:
{state["original_question"]}

Ground Truth Answer:
{known_answer}

Generated Answer:
{state['final_answer']}

You must adhere to the following strict criteria:

1.  **Factual Consistency:** The "Generated Answer" must be factually consistent with the "Ground Truth Answer". It must not contain information that directly contradicts the ground truth.
2.  **Completeness:** The "Generated Answer" must address all parts of the "Query". It is considered "Incorrect" if it omits critical information that is present in the "Ground Truth Answer" and is necessary for a full response.
3.  **Relevance:** The "Generated Answer" must directly answer the user's "Query". An answer that is factually correct but irrelevant to the question is "Incorrect".

**Important Note:** Differences in phrasing, verbosity, or style between the "Generated Answer" and the "Ground Truth Answer" are acceptable as long as the core semantic meaning is the same and the criteria above are met.

Think step-by-step and provide a detailed reasoning process that compares the two answers given the context of the question. Include at least 3 steps in your reasoning, but more as needed.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> correct OR incorrect </answer>
"""
    try:
        if not llm:
            llm = ChatOpenAI(model=state["model"],
                            api_key=state["api_key"],
                            base_url=state["base_url"])
        response = llm.invoke(prompt).content.strip()
        if state["verbose"]:
            print(f"Evaluation response: {response}")
        return 'incorrect' not in response.lower()
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return None


def handle_simple_question(user_input: str, chat_history: list[dict], llm: ChatOpenAI) -> dict:
    # Get the last 4 turns of the chat history
    recent_history = chat_history[-4:]
    
    # Format the chat history for the LLM
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in surgery-related topics. Use the chat history and your knowledge to answer the user's question. If you can't answer based on the given information, say so. If the question is not related to surgery, politely redirect the user to ask a surgery-related question."}
    ]
    
    for turn in recent_history:
        messages.append({"role": "user", "content": turn['user']})
        messages.append({"role": "assistant", "content": turn['bot']})
    
    # Add the current user input
    messages.append({"role": "user", "content": user_input})
    
    # Make the LLM call
    response = llm.invoke(messages).content.strip()
    
    return {
        "final_answer": response,
        "answers": "",
        "wikipedia_results": None,
        "pending_queries": []
    }