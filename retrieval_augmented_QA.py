import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOP_K = 5
PATH_TO_CHUNKS_JSONL = Path("data/chunks.jsonl")
QUERY = "What are the main categories of explainability techniques for large language models?"

def load_chunks(jsonl_path: Path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks = load_chunks(PATH_TO_CHUNKS_JSONL)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    knowledge_base = FAISS.from_texts(texts, embeddings)
    
    if QUERY:
        docs = knowledge_base.similarity_search(QUERY, k=MAX_TOP_K)
        llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.2) # more the temperature, more the creativity of the response, Temperature = 0.2 → keeps answers factual.

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback():

            response = chain.run(input_documents=docs, question=QUERY)
            print(f"Response: {response}")

if __name__ == "__main__":
    main()
