import yaml
import json
from pathlib import Path
from typing import List, Dict
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import streamlit as st

def load_chunks(jsonl_path: Path)-> List[Dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def load_config(path_to_config:str)->Dict:
    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    return config


@st.cache_resource
def build_knowledge_base(chunks: List[Dict], config:Dict):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name=config['EMBEDDINGS_MODEL'])
    return FAISS.from_texts(texts, embeddings)

def get_LLM(config:Dict):
    
    return ChatOpenAI(model_name=config['GPT_MODEL'], temperature=config['TEMPERATURE'])