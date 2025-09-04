import json

llama32_3b_files = [
    "eval_results/surgical_qa_dataset_evaluation_results_llama32-3b_backup.json",
    "eval_results/surgical_qa_dataset_evaluation_results_llama32-3b.json",
    "eval_results/surgical_qa_dataset_evaluation_results_v2_backup.json",
]

llama32_rag_file = "eval_results/vanilla_rag_evaluation_results_llama32.json"
llama32_react_file = "eval_results/react_rag_evaluation_results_llama32.json"

gpt4o_file = "eval_results/qa_results_4o_wo_rag.json"
llama31_file = "eval_results/qa_results_llama31_wo_rag.json"


rep_examples = {}

rep_examples["llama32_deretsyn"] = []
for file in llama32_3b_files:
    with open(file, "r") as f:
        data = json.load(f)
    for packet in data:
        if packet['is_correct'] == True:
            rep_examples["llama32_deretsyn"].append(packet)

rep_examples["llama32_rag"] = []
with open(llama32_rag_file, "r") as f:
    data = json.load(f)
for packet in data:
    if packet['is_correct'] == False:
        rep_examples["llama32_rag"].append(packet)

rep_examples["llama32_react"] = []
with open(llama32_react_file, "r") as f:
    data = json.load(f)
for packet in data:
    if packet['is_correct'] == False:
        rep_examples["llama32_react"].append(packet)

rep_examples["gpt4o"] = []
with open(gpt4o_file, "r") as f:
    data = json.load(f)
for packet in data:
    if packet['is_correct'] == False:
        rep_examples["gpt4o"].append(packet)

rep_examples["llama31"] = []
with open(llama31_file, "r") as f:
    data = json.load(f)
for packet in data:
    if packet['is_correct'] == False:
        rep_examples["llama31"].append(packet)

# find any packets that have the same value in "question" for all the lists
# use sets for this, create new lists with just the question and check intersections with all sets
llama32_derestsyn_qs = set(packet['question'] for packet in rep_examples["llama32_deretsyn"])
llama32_rag_qs = set(packet['question'] for packet in rep_examples["llama32_rag"])
llama32_react_qs = set(packet['question'] for packet in rep_examples["llama32_react"])
gpt4o_qs = set(packet['question'] for packet in rep_examples["gpt4o"])
llama31_qs = set(packet['question'] for packet in rep_examples["llama31"])

# find intersection of all sets
intersection = llama32_derestsyn_qs.intersection(llama32_rag_qs, llama32_react_qs, gpt4o_qs, llama31_qs)
print(len(intersection))

# for each question in the intersection, print the question and the answers from each model and write them to a text file
def get_answer(packet):
    key = "rag_answer"
    if 'rag_answer' not in packet:
        key = "generated_answer"
    return str(packet[key])

with open("representative_examples.txt", "w") as f:
    for question in intersection:
        f.write("Question: " + question + "\n")
        for packet in rep_examples["llama32_deretsyn"]:
            if packet['question'] == question:
                f.write("Known answer: " + packet['known_answer'] + "\n")
                f.write("Llama32 Deretsyn: ")
                f.write(get_answer(packet))
                f.write("\n")
                break
        for packet in rep_examples["llama32_rag"]:
            if packet['question'] == question:
                f.write("Llama32 RAG: ")
                f.write(get_answer(packet))
                f.write("\n")
                break
        for packet in rep_examples["llama32_react"]:
            if packet['question'] == question:
                f.write("Llama32 React: ")
                f.write(get_answer(packet))
                f.write("\n")
                break
        for packet in rep_examples["gpt4o"]:
            if packet['question'] == question:
                f.write("GPT4O: ")
                f.write(get_answer(packet))
                f.write("\n")
                break
        for packet in rep_examples["llama31"]:
            if packet['question'] == question:
                f.write("Llama31: ")
                f.write(get_answer(packet))
                f.write("\n")
                break
        f.write("\n")
