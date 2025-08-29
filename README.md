# Explainable Document-QA Assistant with LLMs

This project implements a **Retrieval-Augmented Question Answering (RAG)** pipeline that allows users to query a collection of PDF documents and get factually grounded answers. It was developed as a portfolio project to demonstrate **LLM application development, information retrieval, and evaluation**.

---

## Features
- Extracts and cleans text from PDFs 
- Splits text into **semantic chunks** for efficient retrieval
- Embeds chunks with **sentence-transformers**
- Stores embeddings in a **FAISS vector database**
- Retrieves the most relevant chunks for a user query
- Uses **OpenAI GPT models** for answer generation
- Provides a **Streamlit UI** for interactive querying
- Evaluates the system with metrics:
  - **Faithfulness**
  - **Answer Relevancy**
  - **Context Precision**
  - **Context Recall**

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/snehabanerjee2017/explainable_document_QA_assistant.git
   cd explainable_document_QA_assistant

2. **Set up environment**

    python3 -m venv venv
    source venv/bin/activate

3. **Install dependencies**

    pip install -r requirements.txt

4. **Set your environment variables**
    Create a .env file in the root directory:

        OPENAI_API_KEY=your_openai_key_here
        HUGGINGFACEHUB_API_TOKEN=your_huggingfacehub_api_token_here

5. **Process PDFs**
    Put your PDFs in the data/ folder, then run:

        python3 data_processing.py

6. **Launch the Document-QA Assistant with Streamlit App**

    streamlit run data_processing.py

    Enter a question in the UI.
    Get an answer sourced from your documents.

7. **Evaluation**
    Evaluate the RAG pipeline with faithfulness, answer relevancy, and context metrics:
    
        python3 evaluation.py

    Example output:
        {'faithfulness': 0.90, 'answer_relevancy': 0.85, 'context_precision': 0.78, 'context_recall': 0.65}
  
**Future Improvements**
- Add support for multi-modal documents (images + text)
- Use reranking models for better retrieval
- Deploy via Docker or cloud services

## Tech Stack
- Text Extraction: Pypdf, langchain-text-splitters
- Embeddings: sentence-transformers, HuggingFace
- Vector DB: FAISS 
- LLM: OpenAI GPT (via HuggingFace)
- Pipeline: LangChain
- Explainability: faithfulness checks, highlighting
- UI: Streamlit 

# Queries for Explainable AI / LLM Papers
“What are the main categories of explainability techniques for large language models?” <br />
→ Should retrieve taxonomy (intrinsic vs post-hoc, attention visualization, feature attribution, etc.). <br />
“How do surveys describe the trade-off between accuracy and interpretability?” <br />
→ Should pull from XAI survey papers. <br />
“Which methods are considered state-of-the-art for explainable vision-language models?” <br />
→ Relevant for multimodal explainability papers. <br />
“How can LLMs themselves be used to generate explanations?” <br />
→ Some surveys mention LLMs as XAI tools. <br />
“What limitations of explainability are identified in recent literature?” <br />
→ Expected answers: lack of standard evaluation, risk of misleading explanations, scalability issues. <br />

# Queries for Application-Focused Papers
“How can explainable AI help in digital advertising according to recent studies?” <br />
→ Should retrieve from the advertising/XAI paper. <br />
“What role do LLMs play in causal inference explainability?” <br />
→ Pulls from the causal auditor paper. <br />
“What are the challenges in applying XAI for human activity recognition?” <br />
→ Should retrieve from the activity recognition paper. <br />

# Queries for EU AI Act (Policy/Regulation)
“What are the four risk categories defined in the EU AI Act?” <br />
→ Unacceptable, high-risk, limited-risk, minimal-risk. <br />
“What transparency obligations are imposed on providers of generative AI systems?” <br />
→ Should retrieve obligations like labeling AI-generated content, documentation requirements. <br />
“How does the EU AI Act address high-risk AI systems in healthcare?” <br />
→ Relevant section in Act + summaries from KPMG/Futurium docs. <br />
“What penalties can organizations face for non-compliance with the EU AI Act?” <br />
→ Fines, restrictions, etc.