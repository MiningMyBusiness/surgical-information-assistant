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
        verbose=True
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
        # TODO: update the user's input to reflect context from the chat history
        # TODO: determine whether the user input should be processed with RAG or just answered easily.

    response = run_agents(user_input)
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
