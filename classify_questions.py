import os
import json
import pandas as pd
import argparse
from transformers import pipeline

# Initialize the zero-shot classification pipeline
print("Initializing zero-shot classification model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") # Automatically uses GPU if available, otherwise CPU
print("Model initialized.")

CANDIDATE_LABELS = ['fetal-maternal pathology', 'Cesarean delivery', 'herniorrhaphy']
CLASSIFICATION_THRESHOLD = 0.9

def classify_question(question):
    """Classifies a question using zero-shot classification and returns multiple labels if scores are above a threshold."""
    if not isinstance(question, str) or not question.strip():
        return 'other'

    result = classifier(question, CANDIDATE_LABELS, multi_label=True)

    # Filter labels that meet the threshold
    passing_labels = [
        label for label, score in zip(result['labels'], result['scores'])
        if score > CLASSIFICATION_THRESHOLD
    ]

    if not passing_labels:
        # If no label passes the threshold, fall back to the one with the highest score
        return 'other'

    return '; '.join(passing_labels)

def process_surgical_qa(filepath):
    """Processes the surgical QA dataset."""
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    for item in data:
        item['zero_shot_category'] = classify_question(item.get('question', ''))

    output_path = filepath.replace('.json', '_zero_shot_classified.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved classified data to {output_path}")

def process_csv_data(filepath, column_name):
    """Processes a generic CSV dataset."""
    print(f"Processing {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in {filepath}")
        return

    df['zero_shot_category'] = df[column_name].apply(classify_question)
    output_path = filepath.replace('.csv', '_zero_shot_classified.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved classified data to {output_path}")

def main():
    """Main function to run the classification on all specified files."""
    # Process surgical QA dataset
    process_surgical_qa('surgical_qa_dataset_2025oct_2_cleaned.json')

    # Process SCT dataset
    process_csv_data('sct_data/sct_cleaned_full.csv', 'sct_stem')

    # Process medical exam datasets
    medical_exams_dir = 'medical_exams'
    exam_files = [
        'general_surgery.csv',
        'internal_medicine.csv',
        'obgyn.csv',
        'pediatrics.csv',
        'psychiatry.csv'
    ]

    for filename in exam_files:
        filepath = os.path.join(medical_exams_dir, filename)
        process_csv_data(filepath, 'question')

if __name__ == '__main__':
    main()
