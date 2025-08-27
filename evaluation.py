from retrieval_augmented_QA import load_chunks, build_knowledge_base
from utils import load_config
import yaml

import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback

# For evaluation
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

load_dotenv()
data = load_config("./configs/config.yaml")

def run_evaluation(queries, gold_answers, kb):
    llm = ChatOpenAI(model_name=data['GPT_MODEL'], temperature=data['TEMPERATURE'])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=kb.as_retriever(search_kwargs={"k": data['TOP_K']}),
        memory=memory,
        return_source_documents=True,
    )

    results = []
    total_tokens = 0
    total_time = 0.0

    for query, gold in zip(queries, gold_answers):
        start = time.time()
        with get_openai_callback() as cb:
            response = qa_chain({"question": query})
        end = time.time()

        total_tokens += cb.total_tokens
        total_time += (end - start)

        results.append({
            "question": query,
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response["source_documents"]],
            "ground_truth": gold
        })

    # Convert to HF Dataset
    dataset = Dataset.from_list(results)

    # Evaluate with ragas
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    eval_results = evaluate(dataset, metrics)

    print("\n📊 Evaluation Results:")
    print(eval_results)

    print(f"Avg latency: {total_time/len(queries):.2f} sec/query")
    print(f"Avg tokens: {total_tokens/len(queries):.1f} tokens/query")

    return eval_results


def main():
    chunks = load_chunks(data['PATH_TO_CHUNKS_JSONL'])
    knowledge_base = build_knowledge_base(chunks)

    # Example test set
    queries = [
        "What are the main categories of explainability techniques?",
        "Explain surrogate models in simple terms.",
    ]
    gold_answers = [
        "Feature attribution, surrogate models, and example-based explanations.",
        "Surrogate models are simplified models that mimic the behavior of complex models for interpretability."
    ]

    run_evaluation(queries, gold_answers, knowledge_base)


if __name__== "__main__":
    main()