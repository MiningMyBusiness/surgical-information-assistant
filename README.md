# Surgical Information Assistant

An implementation of a surgical information assistant based on documents from the Vanderbilt University Open Manual of Surgery.

![Animated GIF Placeholder](figures/example_usage.gif)

## Motivation

The Surgical Information Assistant is a tool designed help people access and utilize surgical information. By leveraging the documentation provided in the comprehensive [Open Manual of Surgery in Resource-Limited Settings](https://www.vumc.org/global-surgical-atlas/about), this assistant provides instant, accurate, and context-aware responses to surgical queries. If a question cannot be answered by the documents in the Open Manual of Surgery, then the system falls back upon Wikipedia and synthesizes information across pages to answer the query with a best effort model (the user is notified when this happens).

Key benefits include:
- Rapid access to critical surgical information
- Improved learning experience for students and access to medical information
- Synthesis of information across multiple verified documents (and Wikipedia)

This project aims to bridge the gap between vast amount of medical documentation and its immediate accessibility, making it an invaluable resource as a reference.

NOTE: This project is not associated with or endorsed by the Open Manual of Surgery in Resource-Limited Settings. It uses it as a reference and is meant to build upon it within the scope of the Creative Commons licensing structure.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Retrieval Process](#retrieval-process)
- [Wikipedia as Backup](#wikipedia-as-backup)
- [Contributing](#contributing)
- [License](#license)

## Setup and Installation

### Prerequisites

- Python 3.10+
- pip
- Git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/kbhattacha/surgical-information-assistant.git
   cd surgical-information-assistant
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Set up environment variables:
    Create a .env file in the root directory with the following content:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    COLBERTV2_URL=http://20.102.90.50:2017/wiki17_abstracts
5. Set up Streamlit secrets:
    Create a .streamlit/secrets.toml file with the following content:
    ```bash
    [openai]
    api_key = "your_openai_api_key_here"
6. Download necessary NLTK data:
    ```bash
    python -c "import nltk; nltk.download('punkt')"

## Usage

To run the Surgical Information Assistant:
    ```bash
    streamlit run app.py

Navigate to the provided local URL (usually http://localhost:8501) in your web browser to interact with the assistant.

## Agentic Retrieval

### Decompose, retrieve, synthesize (repeat)
The Surgical Information Assistant uses a sophisticated retrieval process to provide accurate and relevant information:
1. Decompose: The user's query is analyzed and processed to extract sub-questions.
2. Retrieve: Each sub-question is used to search a pre-built FAISS (Facebook AI Similarity Search) index. This index contains vector representations of the Vanderbilt University Open Manual of Surgery content, allowing for fast and efficient similarity-based retrieval. An LLM attemps to answer each sub-question with the associated context that is retrieved from the text index.
3. Synthesize: An LLM then attempts to answer the original question based on each sub-question and it's answer. If it cannot, then it generates new sub-questions for retrieval (go back to Step 2). If it can answer the original question, then it generates an answer and some follow-up questions.
4. If Steps 2 and 3 have been repeated 3 times, then the system stops and moves on to using Wikipedia.

### Wikipedia as Backup
When the primary surgical database doesn't yield satisfactory results, the assistant falls back to Wikipedia as a secondary source:
1. Fast Search: Initially, a fast search is performed using ColBERTv2, querying a pre-indexed Wikipedia abstract dataset.
2. Slow Search: If needed, a more comprehensive search is conducted using the wikipedia Python library, which retrieves full Wikipedia page content.
3. Integration: Results from both fast and slow searches are combined with previous information from primary surgical texts (if that exists) to provide a comprehensive backup answer.
4. LLM Processing: The Wikipedia results are processed by the LLM to ensure relevance and coherence with the original query.

This dual-search approach ensures that users receive informative responses even for queries not directly covered in the surgical manual.

## Contributing
Contributions to the Surgical Information Assistant are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (git checkout -b feature/AmazingFeature)
3. Make your changes
4. Commit your changes (git commit -m 'Add some AmazingFeature')
5. Push to the branch (git push origin feature/AmazingFeature)
6. Open a Pull Request

## License
This project is licensed under the CC0 1.0 Universal - see the LICENSE file for details.

## Citation

If you use this project in your research or work, please cite it using the following BibTeX entry:

```bibtex
@misc{surgical_information_assistant_codebase,
  author = {Bhattacharya, Kiran},
  title = {Surgical Information Assistant},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kbhattacha/surgical-information-assistant}},
}
