import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from config import Config
from chroma_db import ChromaHistoryDB

cfg = Config()

C_ap = 1e12
d_attn = cfg.EMBEDDING_DIM
d_ffn = cfg.ATTENTION_FFN_DIM
T_win = 10
phi = 0.1
alpha = cfg.ALPHA
beta = cfg.BETA
epsilon_attn = cfg.ATTENTION_WEIGHT_THRESHOLD

tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
if cfg.SEP_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.SEP_TOKEN]})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

embedding_model = AutoModel.from_pretrained(cfg.EMBEDDING_MODEL).to(cfg.DEVICE)
embedding_model.resize_token_embeddings(len(tokenizer))
embedding_model.eval()

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(
            d_model=d_attn,
            nhead=8,
            dim_feedforward=d_ffn,
            activation="relu",
            batch_first=True
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class AttentionRefinementModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer() for _ in range(cfg.ATTENTION_LAYERS)
        ])
        self.all_attention_weights = []

    def forward(self, src, src_key_padding_mask=None):
        self.all_attention_weights = []
        output = src
        for layer in self.layers:
            output, attn_weights = layer(output, src_key_padding_mask=src_key_padding_mask)
            self.all_attention_weights.append(attn_weights)
        return output

attention_model = AttentionRefinementModel().to(cfg.DEVICE)

def load_user_dataset(file_path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_parquet(file_path)
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_id": str(row[cfg.USER_ID_COL]),
            "user_question": str(row[cfg.QUERY_COL]),
            "model_answer": str(row[cfg.RESPONSE_COL]),
            "turn_number": int(row[cfg.TURN_COL]),
            "tau_i": float(row[cfg.MAX_DELAY_COL]),
            "a_i": float(row.get("a_i", 0.8))
        })
    return dataset

def split_dataset(dataset, history_ratio=0.4):
    user_data = {}
    for data in dataset:
        user_id = data["user_id"]
        if user_id not in user_data:
            user_data[user_id] = []
        user_data[user_id].append(data)

    history_data = []
    test_data = []

    for user_id, data_list in user_data.items():
        data_list.sort(key=lambda x: x["turn_number"])
        total_turns = len(data_list)
        history_size = int(total_turns * history_ratio)
        history_data.extend(data_list[:history_size])
        test_data.extend(data_list[history_size:])

    return history_data, test_data

def analyze_user_turn_statistics(dataset):
    user_stats = {}
    for data in dataset:
        user_id = data["user_id"]
        turn = data["turn_number"]
        if user_id not in user_stats:
            user_stats[user_id] = {"max_turn": turn, "turns": []}
        user_stats[user_id]["max_turn"] = max(user_stats[user_id]["max_turn"], turn)
        user_stats[user_id]["turns"].append(turn)

    sorted_users = sorted(user_stats.keys())
    for user_id in sorted_users:
        user_stats[user_id]["turns"].sort()

    return user_stats

def build_history_text(turn, user_question, model_answer):
    return f"Turn {turn}: User asked: {user_question} → Assistant answered: {model_answer}"

def calculate_K_max(
        C_ap: float,
        d_attn: int,
        d_ffn: int,
        L_q_i: int,
        tokenizer: AutoTokenizer,
        top_k_candidates: List[str]
) -> int:
    if not top_k_candidates:
        return 1
    L_R_avg = np.mean([len(tokenizer.encode(text)) for text in top_k_candidates])
    denominator = 4 * L_R_avg * d_attn * (d_attn + d_ffn)
    if denominator == 0:
        return 1
    K_max = int(np.floor(C_ap / denominator) - L_q_i)
    return max(K_max, 1)

def ann_retrieval(
        chroma_db: ChromaHistoryDB,
        current_user_id: str,
        current_question: str,
        current_turn: int,
        tokenizer: AutoTokenizer
) -> Tuple[List[str], int, float]:
    retrieval_result = chroma_db.ann_retrieval_with_time_discount(
        query=current_question,
        current_turn = current_turn,
        user_id=current_user_id,
        top_k=cfg.TOP_K,
        similarity_threshold=0.55
    )

    L_q_i = len(tokenizer.encode(current_question))
    K_max = calculate_K_max(
        C_ap, d_attn, d_ffn, L_q_i,
        tokenizer, retrieval_result["top_k_histories"]
    )
    K_i = min(K_max, len(retrieval_result["top_k_histories"]))

    return (
        retrieval_result["top_k_histories"][:K_i],
        K_i,
        retrieval_result["retrieval_time"]
    )

def calculate_matching_score(query, histories, refined_hist_reprs=None):
    with torch.no_grad():
        query_inputs = tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        ).to(cfg.DEVICE)
        query_emb = embedding_model(**query_inputs).last_hidden_state.mean(dim=1)

        if not histories:
            return []

        if refined_hist_reprs is not None:
            hist_embs = refined_hist_reprs
        else:
            hist_inputs = tokenizer(
                histories, padding=True, truncation=True, return_tensors="pt"
            ).to(cfg.DEVICE)
            hist_embs = embedding_model(**hist_inputs).last_hidden_state.mean(dim=1)

        scores = []
        for emb in hist_embs:
            if query_emb.dim() > 2:
                query_emb = query_emb.squeeze()
            if emb.dim() > 1:
                emb = emb.squeeze()
            sim_tensor = torch.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0), dim=1)
            if sim_tensor.numel() == 1:
                sim = sim_tensor.item()
            else:
                sim = sim_tensor.mean().item()
            scores.append(sim)
        return scores

def process_long_history_with_sliding_window(text, tokenizer, model, max_length=512, stride=256):
    try:
        if not isinstance(text, str):
            text = str(text)

        tokens = tokenizer.encode(text, truncation=False)
        results = []

        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i + max_length]
            if len(window_tokens) < max_length:
                window_tokens = window_tokens + [tokenizer.pad_token_id] * (max_length - len(window_tokens))

            with torch.no_grad():
                input_tensor = torch.tensor([window_tokens]).to(cfg.DEVICE)
                output = model(input_tensor)
                results.append(output.last_hidden_state)

        return torch.mean(torch.cat(results, dim=1), dim=1)
    except Exception:
        return torch.zeros(1, model.config.hidden_size).to(cfg.DEVICE)

def select_optimal_histories(question, histories, max_tokens=500):
    try:
        if not isinstance(question, str):
            question = str(question)

        question_tokens = len(tokenizer.encode(question + " " + cfg.SEP_TOKEN))
        available_tokens = max_tokens - question_tokens

        selected_histories = []
        current_tokens = 0
        long_history_repr = None
        has_long_history = False

        for i, history in enumerate(histories):
            if not isinstance(history, str):
                history = str(history)

            history_tokens = len(tokenizer.encode(history + " " + cfg.SEP_TOKEN))

            if current_tokens + history_tokens > available_tokens:
                if i == 0:
                    long_history_repr = process_long_history_with_sliding_window(
                        history, tokenizer, embedding_model
                    )
                    has_long_history = True
                    break
                else:
                    break
            else:
                selected_histories.append(history)
                current_tokens += history_tokens

        return selected_histories, long_history_repr, has_long_history
    except Exception:
        return [], None, False

def smart_window_combination(window_outputs):
    if not window_outputs:
        return None

    weights = []
    n_windows = len(window_outputs)
    for i in range(n_windows):
        weight = np.exp(-0.5 * ((i - (n_windows - 1) / 2) / (n_windows / 4)) ** 2)
        weights.append(weight)

    weights = torch.tensor(weights).to(cfg.DEVICE)
    weights = weights / weights.sum()

    combined = sum(w * output.mean(dim=1) for w, output in zip(weights, window_outputs))
    return combined

def attention_refinement(
        current_question: str,
        top_Ki_history: List[str],
        rho_ap: float = 1e-6
) -> Tuple[List[str], float, float, float, float]:
    if not top_Ki_history:
        return [], 0.0, 0.0, 0.0, 0.0

    selected_histories, long_history_repr, has_long_history = select_optimal_histories(
        current_question, top_Ki_history
    )

    if selected_histories:
        query_inputs = tokenizer(
            current_question,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(cfg.DEVICE)
        with torch.no_grad():
            query_outputs = embedding_model(**query_inputs)
            query_emb = query_outputs.last_hidden_state.mean(dim=1)

        if selected_histories:
            hist_inputs = tokenizer(
                selected_histories,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(cfg.DEVICE)
            with torch.no_grad():
                hist_outputs = embedding_model(**hist_inputs)
                hist_embs = hist_outputs.last_hidden_state.mean(dim=1)
        else:
            hist_embs = torch.empty(0).to(cfg.DEVICE)
        similarities = torch.cosine_similarity(query_emb, hist_embs, dim=1).tolist()

        selected_histories = [
            hist for hist, sim in zip(selected_histories, similarities) if sim > 0.0
        ]
        formatted_histories = [
            f"Turn {idx + 1}: User asked '{hist.split('→')[0].strip().replace('：', ':')}' → Assistant answered '{hist.split('→')[1].strip().replace('：', ':')}'"
            for idx, hist in enumerate(selected_histories)
        ]
        history_context = "\n".join(formatted_histories)
    else:
        history_context = ""

    if not selected_histories and not has_long_history:
        return [], 0.0, 0.0, 0.0, 0.0

    fusion_text = f"{current_question} {cfg.SEP_TOKEN} " + f" {cfg.SEP_TOKEN} ".join(history_context)

    inputs = tokenizer(
        fusion_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(cfg.DEVICE)

    with torch.no_grad():
        embedding_output = embedding_model(**inputs).last_hidden_state

    key_padding_mask = (inputs["attention_mask"] == 0).to(cfg.DEVICE)
    start_time = time.time()

    context_repr = attention_model(embedding_output, key_padding_mask)

    T_attn = time.time() - start_time

    attn_weights_list = attention_model.all_attention_weights
    if not attn_weights_list:
        return selected_histories, 0.0, T_attn, 0.0,0.0

    if cfg.ATTENTION_WEIGHT_AGG_MODE == "mean":
        attn_weights = torch.mean(torch.stack(attn_weights_list), dim=0)
    else:
        attn_weights = torch.max(torch.stack(attn_weights_list), dim=0)[0]

    avg_attn_weights = torch.mean(attn_weights, dim=1)

    sep_token_id = tokenizer.convert_tokens_to_ids(cfg.SEP_TOKEN)
    sep_positions = (inputs["input_ids"] == sep_token_id).nonzero(as_tuple=True)[1]
    sep_positions = torch.clamp(sep_positions, 0, inputs["input_ids"].shape[1] - 1)
    sep_positions, _ = torch.sort(sep_positions)

    if len(sep_positions) < 1:
        return selected_histories, 0.0, T_attn, 0.0,0.0

    query_end = sep_positions[0].item()
    history_weights = []

    for i in range(min(len(sep_positions) - 1, len(selected_histories))):
        start = sep_positions[i].item() + 1
        end = sep_positions[i + 1].item() if (i + 1 < len(sep_positions)) else inputs["input_ids"].shape[1]
        start, end = max(0, start), min(end, inputs["input_ids"].shape[1])

        if start >= end:
            history_weights.append(0.0)
            continue

        hist_weight = avg_attn_weights[0, start:end].mean().item()
        history_weights.append(hist_weight)

    valid_indices = [i for i, w in enumerate(history_weights) if w >= 0.0020]
    valid_histories = [selected_histories[i] for i in valid_indices if i < len(selected_histories)]
    valid_weights = [history_weights[i] for i in valid_indices if i < len(history_weights)]

    if has_long_history and long_history_repr is not None:
        with torch.no_grad():
            query_inputs = tokenizer(
                current_question, padding=True, truncation=True, return_tensors="pt"
            ).to(cfg.DEVICE)
            query_emb = embedding_model(**query_inputs).last_hidden_state.mean(dim=1)

            long_history_sim = torch.cosine_similarity(
                query_emb, long_history_repr, dim=1
            ).item()

        if long_history_sim >= cfg.ATTENTION_WEIGHT_THRESHOLD:
            valid_histories.append("[长历史记录]")
            valid_weights.append(long_history_sim)
            long_history_processed = True
        else:
            long_history_processed = False
    else:
        long_history_processed = False

    if not valid_histories:
        return [], 0.0, T_attn, 0.0, 0.0
    avg_attention_weight = np.mean(valid_weights) if valid_weights else 0.0

    refined_hist_reprs = []
    for i, weight in zip(valid_indices, valid_weights):
        if i >= len(sep_positions) - 1:
            continue

        start = sep_positions[i].item() + 1
        end = sep_positions[i + 1].item() if (i + 1 < len(sep_positions)) else inputs["input_ids"].shape[1]
        start, end = max(0, start), min(end, inputs["input_ids"].shape[1])

        if start >= end:
            continue

        hist_repr = context_repr[:, start:end, :].mean(dim=1)
        refined_hist_reprs.append(hist_repr * weight)

    if long_history_processed:
        refined_hist_reprs.append(long_history_repr * long_history_sim)

    if not refined_hist_reprs:
        return [], 0.0, T_attn, 0.0,0.0

    matching_scores = calculate_matching_score(
        current_question, valid_histories, refined_hist_reprs
    )
    if not matching_scores:
        return [], 0.0, T_attn, 0.0,0.0

    total_weight = sum(valid_weights)
    L_i = sum(w * s for w, s in zip(valid_weights, matching_scores)) / total_weight if total_weight > 0 else 0.0

    num_layers = cfg.ATTENTION_LAYERS
    actual_length = inputs['attention_mask'].sum().item()
    L_total = actual_length
    F_attn_per = 8 * L_total * (d_attn ** 2) + 4 * (L_total ** 2) * d_attn
    F_ffn_per = 4 * L_total * d_attn * d_ffn
    F_attn_total = num_layers * (F_attn_per + F_ffn_per)
    E_attn = rho_ap * F_attn_total

    return valid_histories, L_i, T_attn, E_attn, avg_attention_weight

def history_matching_module(
        dataset: List[Dict]
) -> List[Dict]:
    user_stats = analyze_user_turn_statistics(dataset)

    initial_history_ratio = 0.4
    initial_user_history = {}

    for user_id, stats in user_stats.items():
        max_turn = stats["max_turn"]
        initial_turns = int(max_turn * initial_history_ratio)

        user_data = [d for d in dataset if d["user_id"] == user_id]
        user_data.sort(key=lambda x: x["turn_number"])

        initial_user_history[user_id] = []

        for data in user_data:
            if data["turn_number"] <= initial_turns:
                history_text = build_history_text(
                    data["turn_number"],
                    data["user_question"],
                    data["model_answer"]
                )
                initial_user_history[user_id].append((data["turn_number"], history_text))

    chroma_db = ChromaHistoryDB(tokenizer, embedding_model)
    if chroma_db.collection.count() == 0:
        chroma_db.add_history_to_db(initial_user_history)

    matching_results = []

    for user_id, stats in user_stats.items():
        user_data = [d for d in dataset if d["user_id"] == user_id]
        user_data.sort(key=lambda x: x["turn_number"])

        for data in user_data:
            current_turn = data["turn_number"]
            current_question = data["user_question"]

            initial_turns = int(stats["max_turn"] * initial_history_ratio)
            if current_turn <= initial_turns:
                continue

            top_Ki_history, K_i, T_ret = ann_retrieval(
                chroma_db, user_id, current_question, current_turn, tokenizer
            )

            valid_histories, L_i, T_attn, E_attn, avg_attention_weight = attention_refinement(
                current_question, top_Ki_history
            )

            matching_results.append({
                "user_id": user_id,
                "current_question": current_question,
                "turn_number": current_turn,
                "L_i": L_i,
                "K_i": K_i,
                "T_ret": T_ret,
                "T_attn": T_attn,
                "E_attn": E_attn,
                "avg_attention_weight": avg_attention_weight,
                "valid_histories": valid_histories
            })

            history_text = build_history_text(
                current_turn,
                current_question,
                data["model_answer"]
            )

            new_history = {user_id: [(current_turn, history_text)]}
            chroma_db.add_history_to_db(new_history)

    all_Li = [item["L_i"] for item in matching_results]
    return matching_results