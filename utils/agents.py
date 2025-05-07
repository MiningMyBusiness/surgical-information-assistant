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
    api_key: str=""
    base_url: str=""
    done: bool=False
    wikipedia_results: str=None
    pending_queries: List[str]=[]
    final_answer: str=None
    cot_for_answer: str=None
    verbose: bool=False
    faiss_index_path: str="surgical_faiss_index"
    run_async: bool=False

decomposition_prompt = PromptTemplate.from_template(
    """You are an expert at breaking complex questions into simpler ones. Break the following question into smaller sub-questions:
    
    Question: {question}

    Each sub-question should be independent and answerable on it's own without needing reference to other sub-questions. Think of at least 3 sub-questions but no more than 7.
    
    Think step-by-step and make sure to reason through how break the question in sub-questions. 
    
    Create new sub-questions in the following format but do NOT answer the question. Respond in the following format:
    
    <think> Your reasoning here... </think>
    <sub-question> The first sub-question... </sub-question>
    <sub-question> The second sub-question... </sub-question>
    ...
    <sub-question> The last sub-question... </sub-question>"""
)

def get_llm_object(state: DeRetSynState):
    if "ollama" not in state['api_key']:
        return ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    else:
        return ChatOllama(model=state["model"],
                          api_key=state["api_key"],
                          base_url=state["base_url"],
                          num_ctx=32000)

def agent_a_decompose_question(state: DeRetSynState) -> None:
    llm = get_llm_object(state)
    original_question = state["original_question"]
    prompt = decomposition_prompt.format(question=original_question)
    full_response = llm.invoke(prompt).content.strip()
    sub_questions = full_response.strip().split("<sub-question>")[1:]
    sub_questions = [sub_q.split("</sub-question>")[0].strip() for sub_q in sub_questions]
    if state["verbose"]:
        print(f"Full response for sub-question generation: {full_response}")
        print(f"Initial sub-questions: {sub_questions}")
    state["pending_queries"] = sub_questions


def agent_b_retrieve(state: DeRetSynState) -> None:
    faiss_index_path = state["faiss_index_path"]
    vectorstore = get_default_vectorstore(faiss_index_path)

    queries = state["pending_queries"]
    answers = state.get("answers", "")
    new_answers = []

    for q in queries:  # TODO: make these calls asynchronously
        results = vectorstore.search(q, k=3)
        response, snippets = generate_answer_from_question_and_context(state, q, results)
        answer_text = f"Question: {q}\nAnswer: {response}\n\n\n"
        new_answers.append(answer_text)
    combined_answers = "".join(new_answers)
    if state["verbose"]:
        print(f"New answers: {combined_answers}")
    state["answers"] = answers + combined_answers
    state["pending_queries"] = []


def agent_b_retrieve_async(state: DeRetSynState) -> None:
    faiss_index_path = state["faiss_index_path"]
    queries = state["pending_queries"]
    answers = state.get("answers", "")
    new_answers = []

    async def process_query(q):
        # Create a new vectorstore instance for each query
        vectorstore = get_default_vectorstore(faiss_index_path)
        results = await to_thread(vectorstore.search)(q, k=3)
        response, snippets = await generate_answer_from_question_and_context_async(state, q, results)
        return f"Question: {q}\nAnswer: {response}\n\n\n"
    
    async def run_queries():
        tasks = [process_query(q) for q in queries]
        return await asyncio.gather(*tasks)

    # Create a new event loop and run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        new_answers = loop.run_until_complete(run_queries())
    finally:
        loop.close()

    combined_answers = "".join(new_answers)
    if state["verbose"]:
        print(f"New answers: {combined_answers}")
    state["answers"] = answers + combined_answers
    state["pending_queries"] = []


def get_default_vectorstore(faiss_index_path: str) -> FaissReader:
    return FaissReader(faiss_index_path)


def generate_answer_from_question_and_context(state: DeRetSynState,
                                              question: str,
                                              context: str) -> str:
    llm = get_llm_object(state)
    prompt = f"""Based on the given question and context, generate an answer.
Question: {question}
Context: {context}

Think step-by-step and make sure to reason through how to generate an answer. ONLY rely on the given context to generate the answer. 

Include snippets of the context that support your answer. Do NOT use any information outside of the given context to generate the answer.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer... </answer>
<snippet> First relevant snippet from the context... </snippet>
<snippet> Second relevant snippet from the context... </snippet>
...
<snippet> The last relevant snippet from the context </snippet>"""
    full_response = llm.invoke(prompt).content.strip()
    if state["verbose"]:
        print(f"Generated answer for question and context: {response}")
    response = full_response.split("<answer>")[1].split("</answer>")[0].strip()
    snippets = full_response.split("<snippet>")[1:-1]
    snippets = [snippet.split("</snippet>")[0].strip() for snippet in snippets]
    if state["verbose"]:
        print(f"Full response for generating answer from question and context: {full_response}")
    return response, "\n".join(snippets)


async def generate_answer_from_question_and_context_async(state: DeRetSynState,
                                                          question: str,
                                                          context: str) -> str:
    llm = get_llm_object(state)
    prompt = f"""Based on the given question and context, generate an answer.
Question: {question}
Context: {context}

Think step-by-step and make sure to reason through how to generate an answer. ONLY rely on the given context to generate the answer. 

Include snippets of the context that support your answer. Do NOT use any information outside of the given context to generate the answer.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> The generated answer... </answer>
<snippet> First relevant snippet from the context... </snippet>
<snippet> Second relevant snippet from the context... </snippet>
...
<snippet> The last relevant snippet from the context </snippet>"""
    response = await to_thread(llm.invoke)(prompt)
    content = response.content.strip()
    answer = content.split("<answer>")[1].split("</answer>")[0].strip()
    snippets = content.split("<snippet>")[1:-1]
    snippets = [snippet.split("</snippet>")[0].strip() for snippet in snippets]
    if state["verbose"]:
        print(f"Generated answer for question and context (async): {content}")
    return answer, "\n".join(snippets)


def agent_c_synthesize(state: DeRetSynState) -> None:
    llm = get_llm_object(state)
    original_question = state["original_question"]
    answers = state["answers"]

    check_prompt = f"""
You are a reasoning engine. Given the following sub-question answers, determine whether they are enough to fully answer the original question. ONLY rely on the knowledge to determine whether the question can be answered.

If yes, then provide the answer. Make your answer detailed and structured with sections, as appropriate. Include as much relevant information as possible from the knowledge provided.

If you determine that you cannot answer the original question, then suggest what additional questions should be asked.

Original Question:
{original_question}

Knowledge:
{answers}

Think step-by-step to reason through you answer and consider the relevant information. Respond in the following format:
<think> Your reasoning here... </think>
<can_answer> yes OR no </can_answer>
<answer> The answer to the original question... </answer>
<new_questions> The first new sub-question... </new_questions>
<new_questions> The second new sub-question... </new_questions>
...
<new questions> The last new sub-question </new_questions>
"""
    response = llm.invoke(check_prompt).content.strip()
    if state["verbose"]:
        print(f"Synthesizer response: {response}")

    can_answer = response.split("<can_answer>")[1].split("</can_answer>")[0].strip().lower()
    if can_answer == "yes":
        answer_text = response.split("<answer>")[1].split("</answer>")[0].strip()
        if "new-questions>" in response:
            answer_text = answer_text.split("<new-questions>")[0].strip()
        state["done"] = True
        state["final_answer"] = answer_text
        state["iterations"] = 1
    else:
        state["done"] = False
        state["iterations"] += 1
    new_queries = []
    new_q_block = response.split("<new_questions>")[1:]
    new_queries = [q.split("</new_questions>")[0].strip() for q in new_q_block]
    state["pending_queries"] = new_queries


def agent_d_best_effort(state: DeRetSynState):
    original_question = state["original_question"]
    search_wikipedia(state)
    generate_prompt = f"""
You are a reasoning engine. Given the following original question and sub-question answers, formulate an answer to the best of your ability.

Original Question:
{original_question}

Knowledge:
{state["answers"]}
{state["wikipedia_results"]}

Think step-by-step to reason through you answer and consider the relevant information. Respond in the following format:
<think> Your reasoning here... </think>
<answer> The answer to the original question... </answer>
"""
    llm = get_llm_object(state)
    response = llm.invoke(generate_prompt).content.strip()
    if state["verbose"]:
        print(f"Best-effort response with help from Wikipedia: {response}")
    answer_text = response.split("<answer>")[1].split("</answer>")[0].strip()
    answer_text += "\n\n" + "NOTE: I could not answer the question completely with the available documents. I have tried to use Wikipedia to help me answer the question to the best of my ability."
    state["done"] = True
    state["final_answer"] = answer_text

def search_wikipedia(state: DeRetSynState) -> str:
    results_fast = search_wikipedia_fast(state["original_question"])
    results_slow = search_wikipedia_slow(state["original_question"])
    state["wikipedia_results"] = results_fast + "\n\n" + results_slow

def search_wikipedia_fast(query: str) -> str:
    try:
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
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


def agent_e_follow_up_question_generator(state: DeRetSynState) -> None:
    original_question = state["original_question"]
    final_answer = state["final_answer"]
    prompt = f"""You are a reasoning engine. Given the following original question and final answer, generate 3 follow-up questions that help expand on the original question and the answer in a step-wise manner.

Original Question:
{original_question}

Final Answer:
{final_answer}

Think step-by-step to reason through your answer and consider the relevant information. Respond in the following format:
<think> Your reasoning here... </think>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>"""
    llm = get_llm_object(state)
    response = llm.invoke(prompt).content.strip()
    if state["verbose"]:
        print(f"Follow-up questions response: {response}")
    follow_up_questions = response.split("<follow_up_questions>")[1:-1]
    follow_up_questions = [q.split("</follow_up_questions>")[0].strip() for q in follow_up_questions]
    state["pending_queries"] = follow_up_questions


def agent_f_cot_generator(state: DeRetSynState) -> None:
    prompt = f"""
You are a reasoning engine. Based on the following question and knowledge, provide a detailed, step-by-step reasoning to arrive at an answer. Include at least 3 steps, but more as needed.

Question:
{state["original_question"]}

Knowledge:
{state["answers"]}
{state["wikipedia_results"] if "wikipedia_results" in state else ""}

Provide your response in this format:

<think> Your reasoning here... </think>
<answer> The final answer here... </answer>
"""
    llm = get_llm_object(state)
    response = llm.invoke(prompt).content.strip()
    if state["verbose"]:
        print(f"COT response: {response}")
    cot = response.split("<think>")[1].split("</think>")[0].strip()
    state['cot_for_answer'] = cot


def orchestrator(state: DeRetSynState):
    # Step 1: Decompose the question
    agent_a_decompose_question(state)
    yield {"step": "decompose_complete", "sub_questions": state["pending_queries"]}

    keep_going = True
    while keep_going:
        # Step 2: Retrieve relevant documents
        if state["run_async"]:
            agent_b_retrieve_async(state)
        else:
            agent_b_retrieve(state)
        yield {"step": "retrieve_complete", "answers": state["answers"]}

        # Step 3: Synthesize the answer
        agent_c_synthesize(state)
        yield {"step": "synthesize_complete", "done": state["done"], "final_answer": state.get("final_answer"), "new_queries": state.get("pending_queries")}

        # Check if we are done
        keep_going = not state["done"]

        if state["iterations"] >= 2 and keep_going:
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


def evaluate_answer(state: DeRetSynState, known_answer: str, llm: ChatOpenAI=None) -> bool:
    prompt = f"""You are a medical reasoning engine that compares two answers to a given question to determine whether the answers are the same. Here is the question and the two answers:

Question:
{state["original_question"]}

Answer 1:
{known_answer}

Answer 2:
{state['final_answer']}

Think step-by-step and provide a detailed reasoning process that compares the two answers given the context of the question. Include at least 3 steps in your reasoning, but more as needed.

Respond in the following format:

<think> Your reasoning here... </think>
<answer> TRUE if the answers are similar, FALSE otherwise... </answer>
"""
    try:
        if not llm:
            llm = ChatOpenAI(model=state["model"],
                            api_key=state["api_key"],
                            base_url=state["base_url"])
        response = llm.invoke(prompt).content.strip()
        if state["verbose"]:
            print(f"Evaluation response: {response}")
        return 'true' in response.lower()
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
