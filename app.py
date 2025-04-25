import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.agents import orchestrator, DeRetSynState
import torch
import sys

# Check if the "local" argument is provided
local = len(sys.argv) > 1 and sys.argv[1] == "local"
key_prefix = ""
if local:
    key_prefix = "local_"

# Check if the "async" argument is provided
run_async = len(sys.argv) > 2 and sys.argv[2].lower() == "async"

# fix issue with torch path, in case it occurs with streamlit
torch.classes.__path__ = []

# Set up LLM
llm = ChatOpenAI(model=st.secrets[key_prefix + "model"],
                 api_key=st.secrets[key_prefix + "api_key"],
                 base_url=st.secrets[key_prefix + "base_url"])

# --- Decision Layer --- #
decision_prompt = PromptTemplate.from_template(
    """Is the following question about surgery or about something else?

Think step-by-step and reason through your answer. Respond in the following format: 

<thinking> Your reasoning here... </thinking>
<answer> yes OR no </answer>

Here are some examples:

Question:
What is the primary purpose of the coronary artery bypass graft?
Response:
<thinking> The question is asking about a coronary bypass graft which is related to surgery. So yes, this question is about surgery. </thinking>
<answer> yes </answer>

Question:
Is machine learning useful for solving complex medical problems?
Response:
<thinking> The question is asking about machine learning as a tool for solving complex medical problems. While the question does specify medical problems, it does not refer to surgery or topics related to surgery. </thinking>
<answer> no </answer>

Question:
{question}
Response:
"""
)

def should_use_langgraph(question):
    response = llm.invoke(decision_prompt.format(question=question)).content.strip()
    decision = response.split("<answer>")[1].split("</answer>")[0].strip().lower()
    return "yes" in decision

def update_user_input_with_context(user_input, chat_history):
    # Get the last 4 turns of the chat history
    recent_history = chat_history[-4:]
    
    # Format the chat history for the LLM
    formatted_history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in recent_history])
    
    # Prepare the prompt for the LLM
    prompt = f"""Given the following chat history and a new user input, update the user input to include any relevant context from the chat history. Only add necessary context, and keep the updated input concise.
Think step-by-step and reason through your answer. Respond in the following format:

<thinking> Your reasoning here... </thinking>
<updated_input> Updated user input here... </updated_input>
    
Here are some examples:

CHAT HISTORY:
User: What are the main types of sutures used in surgery?
Assistant: The main types of sutures used in surgery are absorbable and non-absorbable sutures. Absorbable sutures dissolve over time, while non-absorbable sutures need to be removed. Common absorbable sutures include catgut and synthetic materials like polyglycolic acid. Non-absorbable sutures are often made of materials like nylon or silk.
User: How long do absorbable sutures typically last?
Assistant: The duration of absorbable sutures varies depending on the material used. Generally, they can last anywhere from a few days to several months. For example, fast-absorbing sutures may dissolve within 7-10 days, while slower-absorbing sutures can take 60-90 days or even longer.
NEW USER INPUT:
What about non-absorbable ones?
RESPONSE:
<thinking> Considering the previous discussion the user is referring to non-absorbable sutures. Specifically, they are asking about the lifespan or removal time-frame for non-absorbable sutures. I'll restate the question to include this context from the chat history. </thinking>
<updated_input> What is the typical lifespan or removal timeframe for non-absorbable sutures used in surgery? </updated_input>



CHAT HISTORY:
User: What is a laparotomy?
Assistant: A laparotomy is a surgical procedure that involves making a large incision in the abdomen to gain access to the abdominal cavity. It allows surgeons to examine the abdominal organs, diagnose problems, and perform necessary treatments. Laparotomies are used for various purposes, including exploratory surgery, treatment of abdominal trauma, and removal of tumors.
User: Are there any alternatives to this procedure?
Assistant: Yes, there are alternatives to laparotomy, particularly minimally invasive techniques. The main alternative is laparoscopy, also known as keyhole surgery. In laparoscopy, several small incisions are made instead of one large incision. A camera and specialized instruments are inserted through these small incisions to perform the surgery. This technique often results in less pain, faster recovery, and smaller scars compared to traditional laparotomy.

NEW USER INPUT:
What are the risks?

RESPONSE:
<thinking> The user is asking about risks, but it's not clear whether they're referring to laparotomy or laparoscopy, both of which were discussed in the previous messages. Since laparotomy was the original topic and laparoscopy was introduced as an alternative, it would be helpful to ask about the risks of both procedures for a comprehensive answer. </thinking>
<updated_input> What are the risks associated with both laparotomy and laparoscopy procedures? </updated_input>



CHAT HISTORY:
{formatted_history}

New User Input:
{user_input}

RESPONSE:
"""

    # Make the LLM call
    response = llm.invoke(prompt).content.strip()
    answer = response.split("<updated_input>")[1].split("</updated_input>")[0].strip()
    return answer

# --- LangGraph Invocation --- #
def run_agents(user_input):
    initial_state = DeRetSynState(
        original_question=user_input,
        answers="",
        iterations=0,
        faiss_index_path="surgical_faiss_index",
        model=st.secrets[key_prefix + "model"],
        api_key=st.secrets[key_prefix + "api_key"],
        base_url=st.secrets[key_prefix + "base_url"],
        verbose=True,
        run_async=True
    )

    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    progress_placeholder_2 = st.empty()

    with st.spinner("Processing your question..."):
        progress_placeholder.text("Decomposing your question into sub-questions...")
        for result in orchestrator(initial_state):
            if result["step"] == "decompose_complete":
                progress_placeholder.text("Decomposed your question into the following sub-questions.")
                result_placeholder.text("Generated sub-questions:\n- " + "\n- ".join(result["sub_questions"]))
                progress_placeholder_2.text("Retrieving relevant information for each sub-question...")
            
            elif result["step"] == "retrieve_complete":
                progress_placeholder.text("Retrieved relevant information.")
                result_placeholder.text("")
                progress_placeholder_2.text("Synthesizing information...")
            
            elif result["step"] == "synthesize_complete":
                if not result["done"]:
                    progress_placeholder.text("Synthesized information.")
                    result_placeholder.text("Determined that more information is needed. Generated new queries:\n-" + "\n-".join(result["new_queries"]))
                    progress_placeholder_2.text("Retrieving additional information...")
                else:
                    progress_placeholder.text("Synthesized information.")
                    result_placeholder.text("")
                    progress_placeholder_2.text("Processing complete.")
            
            elif result["step"] == "start_best_effort":
                progress_placeholder_2.text("Facing difficulty completely answering the question with available documents. Generating best effort answer using help from Wikipedia...")
            
            elif result["step"] == "best_effort_complete":
                progress_placeholder.text("Best effort answer generated.")
                result_placeholder.text(f"Wikipedia context used for best effort answer:\n{result['wiki_results']}")
            
            elif result["step"] == "final":
                progress_placeholder.text("Final answer generated.")
                result_placeholder.text("")
                progress_placeholder_2.text(f"Processing complete. Number of iterations needed: {result['state']['iterations']}")

    return result["state"]

# --- Streamlit UI --- #
st.set_page_config(layout="wide")

st.title("📚 Chat with the Open Manual of Surgery in Resource-Limited Settings")
st.info("""This app uses Agents and RAG to provide intelligent 
responses to your questions about surgery by using the [Open Manual of Surgery in Resource-Limited Settings](https://www.vumc.org/global-surgical-atlas/about)
created by the Vanderbilt University Medical Center. 
This app can handle complex queries requiring multi-hop reasoning and synthesis across documents.
It can also generate structured long-form responses when asked. The code is open-source 😊 
and follows the same licensing as the documents with the Open Surgical Manual (CC-v1.0 Universal).
DISCLAIMER: This app was not developed by the Vanderbilt University Medical Center. 
Answers are generated by an LLM and LLMs can make mistakes. This app, the code, and the LLM
responses are not endorsed by or associated with the Vanderbilt University Medical Center.""")

if run_async:
    st.info("Running in async mode")
else:
    st.info("Running in sync mode")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show past chat messages
for i,msg in enumerate(st.session_state.chat_history):
    st.chat_message("user").write(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])
        answers = msg.get("answers", "")
        wiki_results = msg.get("wiki_results", "")
        pending_queries = msg.get("pending_queries", [])
        with st.expander("View reference information used to generate the answer"):
            if wiki_results:
                st.write("Wikipedia context used to answer the question:\n" + wiki_results)
            else:
                st.write("Document QA used to answer the question:\n" + answers)
        if pending_queries and i == len(st.session_state.chat_history) - 1:
            with st.expander("Suggested follow-up questions"):
                st.write("\n- " + "\n- ".join(pending_queries))


default_text = "Ask me something about about surgery."
if user_input := st.chat_input(default_text):
    with st.chat_message("user"):
        st.write(user_input)

    if st.session_state.chat_history:
        updated_input = update_user_input_with_context(user_input, st.session_state.chat_history)
    else:
        updated_input = user_input

    if should_use_langgraph(updated_input):
        response = run_agents(updated_input)
    else:
        # Handle non-surgical questions here (you may want to implement a simpler response mechanism)
        response = {"final_answer": "I'm sorry, but I'm specifically designed to answer questions about surgery. Could you please ask a surgery-related question? If you did ask a question related to surgery, then I may have misunderstood your question. Please try again by emphasizing surgery-related topics."}
    with st.chat_message("assistant"):
        st.markdown(response["final_answer"])
        answers = response.get("answers", "")
        wiki_results = response.get("wiki_results", "")
        pending_queries = response.get("pending_queries", [])
        with st.expander("View reference information used to generate the answer"):
            if wiki_results:
                st.write("Wikipedia context used to answer the question:\n" + wiki_results)
            else:
                st.write("Document QA used to answer the question:\n" + answers)
        if pending_queries:
            with st.expander("Suggested follow-up questions"):
                st.write("\n- " + "\n- ".join(pending_queries))

    st.session_state.chat_history.append({"user": user_input, "bot": response["final_answer"],
                                            "answers": response.get("answers", ""),
                                            "wiki_results": response.get("wikipedia_results", None)})
