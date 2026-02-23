from transformers import AutoTokenizer
from config import cfg
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["EVALUATE_OFFLINE"] = "1"

custom_cache_dir = "YOUR_CACHE_DIR"
os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir
os.environ['HF_HOME'] = custom_cache_dir
os.environ['EVALUATE_CACHE'] = custom_cache_dir

embedding_model = SentenceTransformer('YOUR_MODEL_PATH')
kw_model = KeyBERT(model=embedding_model)

tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_PATH")

def calculate_token_length(text: str) -> int:
    if not text:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

from bert_score import score as bert_score

def calculate_bertscore(prediction: str, reference: str, lang: str = "en") -> float:
    if not prediction or not reference:
        return 0.0
    try:
        P, R, F1 = bert_score([prediction], [reference],  lang=lang, verbose=False)
        return F1.mean().item()
    except Exception:
        return fallback_similarity(prediction, reference)

def fallback_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2))