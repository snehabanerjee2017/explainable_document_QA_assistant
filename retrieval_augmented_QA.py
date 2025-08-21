import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

import streamlit as st

load_dotenv()
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOP_K = 5
PATH_TO_CHUNKS_JSONL = Path("data/chunks.jsonl")

def load_chunks(jsonl_path: Path)-> List[Dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

@st.cache_resource
def build_knowledge_base(chunks: List[Dict]):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return FAISS.from_texts(texts, embeddings)

def main():
    st.set_page_config(page_title="📄 Explainable AI Q&A Assistant", page_icon="🤖", layout="centered")
    st.title("📄 Explainable AI Q&A Assistant")

    chunks = load_chunks(PATH_TO_CHUNKS_JSONL)
    knowledge_base = build_knowledge_base(chunks)

    query = st.text_input("Ask a question about the documents:", placeholder="e.g. What are the main categories of explainability techniques for large language models?")
    
    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            docs = knowledge_base.similarity_search(query, k=MAX_TOP_K)
            llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.2)

            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback():
                response = chain.run(input_documents=docs, question=query)

        # Show answer
        st.subheader("Answer:")
        st.write(response)

        # Show retrieved docs
        with st.expander("Show Retrieved Documents"):
            for doc in docs:
                for chunk in chunks:
                    if chunk["text"] == doc.page_content:
                        st.markdown(f"**Filename:** {chunk['filename'][:-4].replace('_', ' ').title()}")
                        st.markdown(f"{doc.page_content[:500]}...")
                        break
                

if __name__ == "__main__":
    main()
