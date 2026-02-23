import os
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config import cfg
from chroma_db import ChromaHistoryDB
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        raise

def analyze_user_statistics(df: pd.DataFrame) -> Dict:
    user_stats = {}
    for _, row in df.iterrows():
        user_id = str(row[cfg.USER_ID_COL])
        turn = int(row[cfg.TURN_COL])

        if user_id not in user_stats:
            user_stats[user_id] = {"max_turn": turn, "turns": []}

        user_stats[user_id]["max_turn"] = max(user_stats[user_id]["max_turn"], turn)
        user_stats[user_id]["turns"].append(turn)

    for user_id in user_stats:
        user_stats[user_id]["turns"].sort()

    return user_stats

def build_history_text(turn: int, question: str, answer: str) -> str:
    return f"Turn {turn}: User asked: {question} → Assistant answered: {answer}"

def initialize_chromadb(df: pd.DataFrame, history_ratio: float = 0.4) -> ChromaHistoryDB:
    tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
    embedding_model = AutoModel.from_pretrained(cfg.EMBEDDING_MODEL).to(cfg.DEVICE)
    chroma_db = ChromaHistoryDB(tokenizer, embedding_model)

    try:
        chroma_db.client.delete_collection(name=cfg.CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    chroma_db.collection = chroma_db.client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION_NAME,
        embedding_function=chroma_db.embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    user_stats = analyze_user_statistics(df)
    initial_history = {}

    for user_id, stats in user_stats.items():
        max_turn = stats["max_turn"]
        initial_turns = int(max_turn * history_ratio)

        user_data = df[df[cfg.USER_ID_COL] == user_id]
        user_data = user_data.sort_values(by=cfg.TURN_COL)

        initial_history[user_id] = []
        for _, data in user_data.iterrows():
            turn = int(data[cfg.TURN_COL])
            if turn <= initial_turns:
                history_text = build_history_text(
                    turn,
                    str(data[cfg.QUERY_COL]),
                    str(data[cfg.RESPONSE_COL])
                )
                initial_history[user_id].append((turn, history_text))

    chroma_db.add_history_to_db(initial_history)
    return chroma_db

def select_test_queries(df: pd.DataFrame, chroma_db: ChromaHistoryDB, num_queries: int = 30) -> List[Dict]:
    user_stats = analyze_user_statistics(df)
    candidate_queries = []

    for user_id, stats in user_stats.items():
        max_turn = stats["max_turn"]
        initial_turns = int(max_turn * 0.4)

        user_data = df[df[cfg.USER_ID_COL] == user_id]
        user_data = user_data.sort_values(by=cfg.TURN_COL)

        for _, data in user_data.iterrows():
            turn = int(data[cfg.TURN_COL])
            if turn == initial_turns + 1:
                candidate_queries.append({
                    "user_id": user_id,
                    "turn_number": turn,
                    "user_question": str(data[cfg.QUERY_COL]),
                    "model_answer": str(data[cfg.RESPONSE_COL]),
                    "tau_i": float(data[cfg.MAX_DELAY_COL]),
                    "a_i": float(data.get("a_i", 0.8))
                })
                break

    if len(candidate_queries) > num_queries:
        test_queries = random.sample(candidate_queries, num_queries)
    else:
        test_queries = candidate_queries

    return test_queries

def main():
    df = load_dataset("data/multiwoz22_converted.parquet")
    chroma_db = initialize_chromadb(df)
    test_queries = select_test_queries(df, chroma_db, num_queries=50)

    test_queries_df = pd.DataFrame(test_queries)
    test_queries_df.to_parquet("data/test_queries.parquet", index=False)

    return chroma_db, test_queries

if __name__ == "__main__":
    chroma_db, test_queries = main()