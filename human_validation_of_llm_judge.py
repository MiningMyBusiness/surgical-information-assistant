import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="LLM Judge Validation Dashboard",
    page_icon="🔍",
    layout="wide"
)

def load_evaluation_files() -> Dict[str, str]:
    """Load all JSON files from eval_results directory"""
    eval_results_dir = Path("eval_results")
    json_files = {}
    
    if eval_results_dir.exists():
        for file_path in eval_results_dir.glob("*.json"):
            json_files[file_path.name] = str(file_path)
    
    return json_files

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def save_human_ratings(file_path: str, data: List[Dict]):
    """Save updated data with human ratings back to file"""
    try:
        # Create backup
        backup_path = file_path.replace('.json', '_backup.json')
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def get_summary_stats(data: List[Dict]) -> Dict:
    """Calculate summary statistics"""
    total = len(data)
    if total == 0:
        return {"total": 0, "llm_correct": 0, "llm_incorrect": 0, "human_rated": 0}
    
    llm_correct = sum(1 for item in data if item.get('is_correct', False))
    llm_incorrect = total - llm_correct
    human_rated = sum(1 for item in data if 'human_rating' in item)
    
    return {
        "total": total,
        "llm_correct": llm_correct,
        "llm_incorrect": llm_incorrect,
        "human_rated": human_rated
    }

def main():
    st.title("🔍 LLM Judge Validation Dashboard")
    st.markdown("---")
    
    # Sidebar for file selection and filters
    with st.sidebar:
        st.header("📁 File Selection")
        
        # Load available files
        json_files = load_evaluation_files()
        
        if not json_files:
            st.error("No JSON files found in eval_results directory")
            return
        
        selected_file = st.selectbox(
            "Select evaluation file:",
            options=list(json_files.keys()),
            help="Choose a JSON file from the eval_results directory"
        )
        
        if selected_file:
            file_path = json_files[selected_file]
            data = load_json_data(file_path)
            
            if not data:
                st.error("No data loaded from selected file")
                return
            
            # Display summary statistics
            stats = get_summary_stats(data)
            st.markdown("### 📊 Summary Statistics")
            st.metric("Total Questions", stats["total"])
            st.metric("LLM Judge: Correct", stats["llm_correct"])
            st.metric("LLM Judge: Incorrect", stats["llm_incorrect"])
            st.metric("Human Rated", stats["human_rated"])
            
            # Filters
            st.markdown("### 🔍 Filters")
            
            filter_by_llm_judge = st.selectbox(
                "Filter by LLM Judge Result:",
                options=["All", "Correct Only", "Incorrect Only"],
                help="Filter questions based on LLM judge determination"
            )
            
            filter_by_human_rating = st.selectbox(
                "Filter by Human Rating:",
                options=["All", "Rated Only", "Unrated Only"],
                help="Filter questions based on human rating status"
            )
            
            # Apply filters
            filtered_data = data.copy()
            
            if filter_by_llm_judge == "Correct Only":
                filtered_data = [item for item in filtered_data if item.get('is_correct', False)]
            elif filter_by_llm_judge == "Incorrect Only":
                filtered_data = [item for item in filtered_data if not item.get('is_correct', False)]
            
            if filter_by_human_rating == "Rated Only":
                filtered_data = [item for item in filtered_data if 'human_rating' in item]
            elif filter_by_human_rating == "Unrated Only":
                filtered_data = [item for item in filtered_data if 'human_rating' not in item]
            
            st.metric("Filtered Results", len(filtered_data))
    
    # Main content area
    if 'filtered_data' in locals() and filtered_data:
        # Pagination
        items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=1)
        total_pages = (len(filtered_data) - 1) // items_per_page + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.number_input(
                f"Page (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1
            )
        
        # Calculate start and end indices
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_data))
        
        # Display current page items
        st.markdown(f"### Showing items {start_idx + 1}-{end_idx} of {len(filtered_data)}")
        
        # Track if any changes were made
        changes_made = False
        
        for i in range(start_idx, end_idx):
            item = filtered_data[i]
            original_idx = data.index(item)  # Get original index in full dataset
            
            with st.expander(f"Question {i + 1}: {item['question'][:100]}..."):
                # Display question and answers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Question:**")
                    st.write(item['question'])
                    
                    st.markdown("**✅ Ground Truth Answer:**")
                    st.write(item['known_answer'])
                    
                    st.markdown("**🤖 LLM Judge Result:**")
                    judge_result = "✅ Correct" if item.get('is_correct', False) else "❌ Incorrect"
                    st.write(judge_result)
                
                with col2:
                    st.markdown("**🎯 Generated Answer:**")
                    st.write(item['rag_answer'])
                    
                    # Show context if available (using text_area instead of nested expander)
                    if 'document_context' in item and item['document_context']:
                        st.markdown("**📚 Document Context:**")
                        context_text = item['document_context'][:1000] + "..." if len(item['document_context']) > 1000 else item['document_context']
                        st.text_area(
                            "Document Context",
                            value=context_text,
                            height=150,
                            key=f"context_{original_idx}",
                            label_visibility="collapsed"
                        )
                    
                    # Show chain of thought if available (using text_area instead of nested expander)
                    if 'cot' in item and item['cot']:
                        st.markdown("**🧠 Chain of Thought:**")
                        cot_text = item['cot'][:1000] + "..." if len(item['cot']) > 1000 else item['cot']
                        st.text_area(
                            "Chain of Thought",
                            value=cot_text,
                            height=100,
                            key=f"cot_{original_idx}",
                            label_visibility="collapsed"
                        )
                
                # Human rating section
                st.markdown("---")
                st.markdown("**👤 Human Rating:**")
                
                current_human_rating = item.get('human_rating', None)
                current_human_notes = item.get('human_notes', '')
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    human_rating = st.radio(
                        "Is the generated answer correct?",
                        options=[None, True, False],
                        format_func=lambda x: "Not Rated" if x is None else ("✅ Correct" if x else "❌ Incorrect"),
                        index=0 if current_human_rating is None else (1 if current_human_rating else 2),
                        key=f"rating_{original_idx}"
                    )
                
                with col2:
                    human_notes = st.text_area(
                        "Notes (optional):",
                        value=current_human_notes,
                        height=100,
                        key=f"notes_{original_idx}",
                        help="Add any notes about why you rated this way"
                    )
                
                # Update data if rating changed
                if human_rating != current_human_rating or human_notes != current_human_notes:
                    if human_rating is not None:
                        data[original_idx]['human_rating'] = human_rating
                    elif 'human_rating' in data[original_idx]:
                        del data[original_idx]['human_rating']
                    
                    if human_notes.strip():
                        data[original_idx]['human_notes'] = human_notes.strip()
                    elif 'human_notes' in data[original_idx]:
                        del data[original_idx]['human_notes']
                    
                    changes_made = True
                
                # Show LLM judge evaluation if available (using text_area instead of nested expander)
                if 'evaluation' in item and item['evaluation']:
                    st.markdown("**🔍 LLM Judge Evaluation:**")
                    st.text_area(
                        "LLM Judge Evaluation",
                        value=item['evaluation'],
                        height=100,
                        key=f"eval_{original_idx}",
                        label_visibility="collapsed"
                    )
        
        # Save button
        if changes_made:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("💾 Save Changes", type="primary"):
                    if save_human_ratings(file_path, data):
                        st.success("✅ Changes saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save changes")
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_page > 1:
                if st.button("⬅️ Previous Page"):
                    st.rerun()
        
        with col3:
            if current_page < total_pages:
                if st.button("➡️ Next Page"):
                    st.rerun()
    
    else:
        st.info("Select a file and apply filters to view evaluation results.")

if __name__ == "__main__":
    main()