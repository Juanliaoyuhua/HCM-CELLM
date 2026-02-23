import os
import time
from openai import OpenAI
from config import cfg

CLOUD_API_KEY = os.environ.get("YOUR_ENV_VAR_HERE")
if not CLOUD_API_KEY:
    raise ValueError("Missing Cloud API Key")

CLOUD_BASE_URL = "YOUR_CLOUD_BASE_URL"
EDGE_BASE_URL = "YOUR_EDGE_BASE_URL"

EDGE_API_KEYS = [
    'YOUR_EDGE_API_KEY_1',
    'YOUR_EDGE_API_KEY_2',
    'YOUR_EDGE_API_KEY_3'
]

class DeepSeekClient:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=CLOUD_BASE_URL,
            api_key=CLOUD_API_KEY
        )

    def chat_completions(self, messages, temperature=0.7, max_tokens=200, stream=False):
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            inference_time = time.perf_counter() - start_time

            if stream:
                return response, inference_time
            else:
                return {
                    "choices": response.choices,
                    "inference_time": inference_time
                }
        except Exception:
            return {
                "choices": [],
                "inference_time": time.perf_counter() - start_time
            }

class ModelScopeEdgeClient:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.modelscope_id = model_name

        self.client = OpenAI(
            base_url=EDGE_BASE_URL,
            api_key=api_key
        )

        self.extra_body = {
            "enable_thinking": False,
        }

    def chat_completions(self, messages, temperature=0.7, max_tokens=200, stream=False):
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.modelscope_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_body=self.extra_body
            )

            inference_time = time.perf_counter() - start_time

            if stream:
                return response, inference_time
            else:
                return {
                    "choices": response.choices,
                    "inference_time": inference_time
                }
        except Exception:
            return {
                "choices": [],
                "inference_time": time.perf_counter() - start_time
            }

cloud_client = DeepSeekClient(cfg.CLOUD_MODEL)
edge_clients = {}
for idx, model_name in enumerate(cfg.EDGE_MODELS):
    if idx < len(EDGE_API_KEYS):
        api_key = EDGE_API_KEYS[idx]
        edge_clients[model_name] = ModelScopeEdgeClient(model_name, api_key)
    else:
        raise ValueError(f"Missing API keys")