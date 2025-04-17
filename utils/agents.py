from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List
from utils.write_to_milvus import MilvusClient
import dspy
from typing import TypedDict, List
from utils.wikipedia_helps import grab_wikipedia_context
import os

# Define the structure of your state
class DeRetSynState(TypedDict):
    original_question: str=""
    answers: str=""
    iterations: int=0
    collection_name: str="surgical_information",
    model: str=""
    api_key: str=""
    base_url: str=""
    done: bool=False
    wikipedia_results: str=None
    pending_queries: List[str]=[]
    final_answer: str=None
    cot_for_answer: str=None
    verbose: bool=False
    milvus_directory: str=None

decomposition_prompt = PromptTemplate.from_template(
    """You are an expert at breaking complex questions into simpler ones. Break the following question into smaller sub-questions:
    
    Question: {question}

    Each sub-question should be independent and answerable on it's own without needing reference to other sub-questions. Think of at least 3 sub-questions but no more than 10.
    
    Think step-by-step and make sure to reason through how break the question in sub-questions. 
    
    Create new sub-questions in the following format but do NOT answer the question. Respond in the following format:
    
    <thinking> Your reasoning here... </thinking>
    <sub-question> The first sub-question... </sub-question>
    <sub-question> The second sub-question... </sub-question>
    ...
    <sub-question> The last sub-question... </sub-question>"""
)

def agent_a_decompose_question(state: DeRetSynState) -> None:
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    original_question = state["original_question"]
    prompt = decomposition_prompt.format(question=original_question)
    sub_questions = llm.invoke(prompt).content.strip().split("<sub-question>")[1:]
    sub_questions = [sub_q.split("</sub-question>")[0].strip() for sub_q in sub_questions]
    if state["verbose"]:
        print(f"Initial sub-questions: {sub_questions}")
    state["pending_queries"] = sub_questions


def agent_b_retrieve(state: DeRetSynState) -> None:
    collection_name = state["collection_name"]
    milvus_directory = state["milvus_directory"]
    vectorstore = get_default_vectorstore(milvus_directory, collection_name)

    queries = state["pending_queries"]
    answers = state.get("answers", "")
    new_answers = []

    for q in queries:
        results = vectorstore.read_from_milvus(q, k=3)
        response, snippets = generate_answer_from_question_and_context(state, q, results)
        answer_text = f"Question: {q}\nAnswer: {response}\n\n\n"
        new_answers.append(answer_text)
    combined_answers = "".join(new_answers)
    if state["verbose"]:
        print(f"New answers: {combined_answers}")
    state["answers"] = answers + combined_answers
    state["pending_queries"] = []


def get_default_vectorstore(milvus_directory: str, collection_name: str="surgical_information") -> MilvusClient:
    return MilvusClient(collection_name, milvus_directory)


def generate_answer_from_question_and_context(state: DeRetSynState,
                                              question: str,
                                              context: str) -> str:
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    prompt = f"""Based on the given question and context, generate an answer.
Question: {question}
Context: {context}

Think step-by-step and make sure to reason through how to generate an answer. ONLY rely on the given context to generate the answer. 

Include snippets of the context that support your answer. Do NOT use any information outside of the given context to generate the answer.

Respond in the following format:

<thinking> Your reasoning here... </thinking>
<answer> The generated answer... </answer>
<snippet> First relevant snippet from the context... </snippet>
<snippet> Second relevant snippet from the context... </snippet>
...
<snippet> The last relevant snippet from the context </snippet>"""
    response = llm.invoke(prompt).content.strip().split("<answer>")[1].split("</answer>")[0].strip()
    snippets = llm.invoke(prompt).content.strip().split("<snippet>")[1:-1]
    snippets = [snippet.split("</snippet>")[0].strip() for snippet in snippets]
    return response, "\n".join(snippets)


def agent_c_synthesize(state: DeRetSynState) -> None:
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
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
<thinking> Your reasoning here... </thinking>
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
<thinking> Your reasoning here... </thinking>
<answer> The answer to the original question... </answer>
"""
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    response = llm.invoke(generate_prompt).content.strip()
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
<thinking> Your reasoning here... </thinking>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>
<follow_up_questions> follow-up question here... </follow_up_questions>"""
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    response = llm.invoke(prompt).content.strip()
    follow_up_questions = response.split("<follow_up_questions>")[1:-1]
    follow_up_questions = [q.split("</follow_up_questions>")[0].strip() for q in follow_up_questions]
    state["pending_queries"] = follow_up_questions


def agent_f_cot_generator(state: DeRetSynState) -> None:
    prompt = f"""
You are a reasoning engine. Based on the following question and knowledge, provide a detailed, step-by-step reasoning using the knowledge to arrive at an answer. Include at 3 steps but more as needed.

Question:
{state["original_question"]}

Context:
{state["answers"]}
{state["wikipedia_results"]}

Provide your response in this format:

<thinking> Your reasoning here... </thinking>
<answer> The final answer here... </answer>
"""
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    response = llm.invoke(prompt).content.strip()
    cot = response.split("<thinking>")[1].split("</thinking>")[0].strip()
    state['cot_for_answer'] = cot


def orchestrator(state: DeRetSynState):
    # Step 1: Decompose the question
    agent_a_decompose_question(state)
    yield {"step": "decompose_complete", "sub_questions": state["pending_queries"]}

    keep_going = True
    while keep_going:
        # Step 2: Retrieve relevant documents
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


def evaluate_answer(state: DeRetSynState, known_answer: str) -> bool:
    prompt = f"""You are a medical reasoning engine that compares two answers to a given question to determine whether the answers are the same. Here is the question and the two answers:

Question:
{state["original_question"]}

Answer 1:
{known_answer}

Answer 2:
{state['final_answer']}

Think step-by-step and provide a detailed reasoning process that compares the two answers given the context of the question. Include at least 3 steps in your reasoning, but more as needed.

Respond in the following format:

<thinking> Your reasoning here... </thinking>
<answer> TRUE if the answers are similar, FALSE otherwise... </answer>
"""
    llm = ChatOpenAI(model=state["model"],
                     api_key=state["api_key"],
                     base_url=state["base_url"])
    response = llm.invoke(prompt)
    return 'true' in response.content.strip().lower()