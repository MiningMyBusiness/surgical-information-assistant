import json
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Phrases to identify questions that refer directly to the passage
# The phrases are checked in a case-insensitive manner.
FILTER_PHRASES = [
    "who is",
    "the procedure",
    "the passage",
    "this passage",
    "the text",
    "this text",
    "according to the passage",
    "according to the text",
    "what does the passage say",
    "what does the text say",
    "in the provided context",
    "based on the passage",
    "based on the text",
    "from the passage",
    "from the text",
    "the provided document",
    "affiliation"
]

def clean_qa_dataset(input_file, output_file):
    """
    Cleans a QA dataset by removing questions containing specific filter phrases.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file for cleaned data.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data)} records from {input_file}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}")
        return

    cleaned_data = []
    removed_counts = defaultdict(int)
    total_removed = 0

    for item in data:
        question = item.get("question", "").lower()
        is_removed = False
        for phrase in FILTER_PHRASES:
            if phrase in question:
                removed_counts[phrase] += 1
                is_removed = True
        
        if not is_removed:
            cleaned_data.append(item)
        else:
            total_removed += 1

    logging.info(f"Total questions removed: {total_removed}")
    logging.info("Breakdown of removed questions by phrase:")
    for phrase, count in removed_counts.items():
        logging.info(f"  - '{phrase}': {count}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2)
        logging.info(f"Cleaned data saved to {output_file}. Kept {len(cleaned_data)} questions.")
    except IOError as e:
        logging.error(f"Error writing to output file {output_file}: {e}")

if __name__ == "__main__":
    input_dataset = "surgical_qa_dataset_2025oct_2.json"
    output_dataset = "surgical_qa_dataset_2025oct_2_cleaned.json"
    clean_qa_dataset(input_dataset, output_dataset)
