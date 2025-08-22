import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import ConversationalRetrievalChain

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

    # Initialize conversational memory in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Initialize LLM and ConversationalRetrievalChain
    llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.2)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": MAX_TOP_K})
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    # Chat interface
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask me anything about Explainable AI:",
            placeholder="e.g. What are the main categories of explainability techniques?",
            key="user_input",
        )
        submitted = st.form_submit_button("Send")

        docs_and_scores = knowledge_base.similarity_search_with_score(user_input, k=MAX_TOP_K)

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            with get_openai_callback():
                result = conversation_chain(
                    {"question": user_input, "chat_history": st.session_state.chat_history}
                )
            answer = result["answer"]
            st.session_state.chat_history.append((user_input, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Assistant:** {answer}")
            st.markdown("---")

        # Show retrieved docs
        with st.expander("Show Retrieved Documents with Similarity Scores"):
            for (doc, score) in docs_and_scores:
                for chunk in chunks:
                    if chunk["text"] == doc.page_content:
                        similarity = 1 - score
                        st.markdown(f"**Filename:** {chunk['filename'][:-4].replace('_', ' ').title()}, **Similarity Score:** {similarity:.4f}")
                        st.markdown(f"{doc.page_content[:500]}...")
                        break
            

if __name__ == "__main__":
    main()
