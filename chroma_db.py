import time
import torch
import chromadb
from tqdm import tqdm
from typing import List, Dict, Tuple
from config import cfg

class CustomEmbeddingFunction:
    def __init__(self, tokenizer, embedding_model, device):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device

    def name(self):
        return "custom_embedding_function"

    def __call__(self, input: List[str]) -> List[List[float]]:
        with torch.no_grad():
            encoded_input = self.tokenizer(
                input,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            model_output = self.embedding_model(**encoded_input)

            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings.cpu().numpy().tolist()

class ChromaHistoryDB:
    def __init__(self, tokenizer, embedding_model):
        self.client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model.to(cfg.DEVICE)
        self.embedding_model.eval()

        self.embedding_function = CustomEmbeddingFunction(
            tokenizer=self.tokenizer,
            embedding_model=self.embedding_model,
            device=cfg.DEVICE
        )

        self.collection = self.client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_history_to_db(self, user_history: Dict[str, List[Tuple[int, str]]]):
        batch_size = 32
        all_items = []

        for user_id, turns in user_history.items():
            for turn, text in turns:
                existing_ids = self.collection.get(
                    where={"$and": [{"user_id": {"$eq": user_id}}, {"round": {"$eq": turn}}]},
                    include=["metadatas"]
                )["ids"]

                if existing_ids:
                    continue

                all_items.append({
                    "user_id": user_id,
                    "turn": turn,
                    "text": text
                })

        if not all_items:
            return

        for i in tqdm(range(0, len(all_items), batch_size)):
            batch = all_items[i:i + batch_size]
            documents = [item["text"] for item in batch]
            metadatas = [{"user_id": item["user_id"], "round": item["turn"]} for item in batch]
            ids = [f"{item['user_id']}_round_{item['turn']}" for item in batch]

            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception:
                pass

    def ann_retrieval_with_time_discount(self, query: str, current_turn: int, user_id: str,
                                         top_k: int = cfg.TOP_K,
                                         similarity_threshold: float = 0.65) -> Dict:
        if top_k <= 0 or top_k > 100:
            raise ValueError(f"top_k error: {top_k}")
        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Query error")
        start_time = time.time()
        where_clause = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"round": {"$lt": current_turn}}
            ]
        }
        try:
            raw_results = self.collection.query(
                query_texts=[query],
                where=where_clause,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            raw_results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        required_keys = ["documents", "metadatas", "distances"]
        for key in required_keys:
            if key not in raw_results:
                raise KeyError(f"KeyError: {key}")
        if len(raw_results["documents"]) == 0:
            return {
                "top_k_histories": [],
                "top_k_similarities": [],
                "retrieval_time": time.time() - start_time,
                "has_history": False
            }

        query_results = raw_results["documents"][0]
        query_metadatas = raw_results["metadatas"][0]
        query_distances = raw_results["distances"][0]

        ranked_results = []
        for doc, meta, dist in zip(query_results, query_metadatas, query_distances):
            sim_base = 1 - dist
            m = int(meta["round"])
            sim_final = cfg.ALPHA * sim_base + cfg.BETA * m
            ranked_results.append({
                "history_text": doc,
                "round": m,
                "base_similarity": sim_base,
                "final_similarity": sim_final
            })

        high_relevance_results = [r for r in ranked_results if r["final_similarity"] > similarity_threshold]

        if len(high_relevance_results) == 0:
            return {
                "top_k_histories": [],
                "top_k_similarities": [],
                "retrieval_time": time.time() - start_time,
                "has_history": False
            }

        final_results = sorted(high_relevance_results,
                               key=lambda x: x["final_similarity"],
                               reverse=True)

        return {
            "top_k_histories": [r["history_text"] for r in final_results],
            "top_k_similarities": [r["final_similarity"] for r in final_results],
            "retrieval_time": time.time() - start_time,
            "has_history": len(final_results) > 0
        }