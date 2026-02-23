import torch
import numpy as np

class Config:
    USER_ID_COL = "user_id"
    QUERY_COL = "user_question"
    TURN_COL = "turn_number"
    RESPONSE_COL = "model_answer"
    MAX_DELAY_COL = "tau_i"
    REFERENCE_RESPONSE_COL = "model_answer"

    CHROMA_PERSIST_DIR = "YOUR_DB_DIR"
    CHROMA_COLLECTION_NAME = "user_history"
    EMBEDDING_MODEL = "YOUR_EMBEDDING_MODEL_PATH"
    EMBEDDING_DIM = 768

    CLOUD_MODEL = ""
    EDGE_MODELS = [

    ]

    ALPHA = 0.6
    BETA = 0.4
    TOP_K = 10
    ANN_INDEX_TYPE = "hnsw"
    SIMILARITY_THRESHOLD = 0.0

    ATTENTION_HIDDEN_DIM = 256
    ATTENTION_FFN_DIM = 512
    ATTENTION_LAYERS = 5
    ATTENTION_HIDDEN = 1
    SEP_TOKEN = "<SEP>"
    BATCH_SIZE = 8
    EPOCHS = 3
    ATTENTION_WEIGHT_THRESHOLD = 0.00381
    ATTENTION_WEIGHT_AGG_MODE = "mean"

    B_U = 10.0
    p_U = 0.1
    h_U = 0.5
    B_C_TO_E = 8.0
    p_C_TO_E = 0.08
    h_C_TO_E = 0.5
    N0 = 1e-12

    C_ap = 1e12
    d_attn = 768
    d_ffn = 512
    T_win = 10
    phi = 0.1
    rho_ap = 50

    MODEL_PERFORMANCE = {

    }

    EDGE_SERVERS = [

    ]

    MAX_NEW_TOKENS = 200
    TEMPERATURE = 0.3
    REPETITION_PENALTY = 1.2
    MAX_SKETCH_NUM = 4
    MIN_SKETCH_TOKENS = 50
    MAX_SKETCH_TOKENS = 150
    T_TIME_WINDOW = 10
    AVG_SIM = 0.2
    ENERGY_THRESHOLD = 500.0
    CLOUD_POWER = 400
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEFAULT_HISTORY_NUM = 10
    SIMILAR_HISTORY_NUM = 4
    TOPIC_SIMILARITY_THRESHOLD = 0.2
    SEMANTIC_SIMILARITY_THRESHOLD = 0.5
    MAX_HISTORY_LEN = 6
    MAX_HISTORY_TURNS = 2
    HISTORY_EMBEDDING_SIZE = 768
    MAX_SINGLE_HISTORY_TOKENS = 300
    MAX_CONTEXT_LENGTH = 8192

cfg = Config()
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)