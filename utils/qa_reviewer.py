from utils.llms import init_llm

def get_reviewer_prompt(question: str, answer: str, source_chunk: str):
    prompt = f"""Your role is to act as a meticulous evaluator of question-answer (QA) pairs based on a provided document. You will be given a document, a question, and an answer. Your task is to rigorously assess the QA pair against seven specific criteria.

For each of the seven criteria below, you must provide a clear 'Yes' or 'No' answer, followed by a brief, one-sentence justification for your decision.

After evaluating all seven criteria individually, you will provide a final Overall Decision. This final decision must be 'Pass' if and only if the answers to all seven criteria are 'Yes'. If even one criterion is not met, the Overall Decision must be 'Fail'.


##Evaluation Criteria:##
- Answerable from Passage: Is the question fully answerable using only the information present in the passage?
- Question is Self-Referential: Does the question contain phrases that refer directly to the passage itself (e.g., "according to the passage," "in this document")? (Note: A 'No' is desired for this criterion).
- Question is Context-Independent: Is the question completely understandable on its own, without needing to read the passage first?
- Question is Searchable: Is the question clear and specific enough that a person could reasonably use it as a search engine query to find the answer?
- Answer is Complete: Does the provided answer fully and completely address all parts of the question?
- Answer is Grounded: Is all of the information in the answer explicitly supported by the content of the passage?
- Answer is Self-Contained: Does the answer make sense on its own without assuming the reader has context from the passage that isn't already in the question?

Respond in the following format:
<think> You reasoning here... </think>
<answerable_from_passage> Yes/No </answerable_from_passage>
<self_referential> Yes/No </self_referential>
<context_independent> Yes/No </context_independent>
<searchable> Yes/No </searchable>
<complete> Yes/No </complete>
<grounded> Yes/No </grounded>
<self_contained> Yes/No </self_contained>
<overall_decision> Pass/Fail </overall_decision>

##Document:##
{source_chunk}

##Question:##
{question}

##Answer:##
{answer}

##Your Response:##
"""

    return prompt


def parse_response(response: str):
    try:
        answerable_from_passage = response.split('<answerable_from_passage>')[1].split('</answerable_from_passage>')[0]
        self_referential = response.split('<self_referential>')[1].split('</self_referential>')[0]
        context_independent = response.split('<context_independent>')[1].split('</context_independent>')[0]
        searchable = response.split('<searchable>')[1].split('</searchable>')[0]
        complete = response.split('<complete>')[1].split('</complete>')[0]
        grounded = response.split('<grounded>')[1].split('</grounded>')[0]
        self_contained = response.split('<self_contained>')[1].split('</self_contained>')[0]
        overall_decision = response.split('<overall_decision>')[1].split('</overall_decision>')[0]
        return {
            'answerable_from_passage': answerable_from_passage,
            'self_referential': self_referential,
            'context_independent': context_independent,
            'searchable': searchable,
            'complete': complete,
            'grounded': grounded,
            'self_contained': self_contained,
            'overall_decision': overall_decision
        }
    except Exception as e:
        logging.error(f"Error parsing response: {str(e)}")
        return {
            'answerable_from_passage': 'N/A',
            'self_referential': 'N/A',
            'context_independent': 'N/A',
            'searchable': 'N/A',
            'complete': 'N/A',
            'grounded': 'N/A',
            'self_contained': 'N/A',
            'overall_decision': 'N/A'
        }

def review_qa_pair(question: str, answer: str, source_chunk: str):
    llm = init_llm('azure-gpt35')
    prompt = get_reviewer_prompt(question, answer, source_chunk)
    response = llm.invoke(prompt).content
    parsed_response = parse_response(response)
    if parsed_response['overall_decision'] == 'N/A':  # try again once
        response = llm.invoke(prompt).content
        parsed_response = parse_response(response)
    return parsed_response
    