import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from config import cfg
from init_LLM import cloud_client, edge_clients
from utils import calculate_token_length, calculate_bertscore

logger = logging.getLogger(__name__)


def clean_final_response(text):
    pattern_perspective = r'\[\s*Perspective\s+\d+\s+-\s+[^\]]+\]\s*:'
    cleaned_text = re.sub(pattern_perspective, '', text)
    cleaned_text = re.sub(r'\n\s*\[?[A-Za-z]+-[^:\]]+\]?\s*:', '', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()


@dataclass
class InferenceResult:
    content: str
    latency: float
    energy: float
    model_name: str
    sketch_id: int = None


class CloudEdgeInferenceSystem:
    def __init__(self):
        self.cloud_model = cfg.CLOUD_MODEL
        self.edge_models = cfg.EDGE_MODELS
        self.edge_servers = cfg.EDGE_SERVERS
        self.sketch_count = min(cfg.MAX_SKETCH_NUM, len(cfg.EDGE_SERVERS))
        self.cloud_params = cfg.MODEL_PERFORMANCE[cfg.CLOUD_MODEL]
        self.edge_params = {model: cfg.MODEL_PERFORMANCE[model] for model in cfg.EDGE_MODELS}
        self.epsilon_discount = 33.5144

    def _calculate_transmission_metrics(self, data_size: int, is_uplink: bool = True) -> Tuple[float, float]:
        if is_uplink:
            bandwidth = cfg.B_U
            power = cfg.p_U
            channel_gain = cfg.h_U
        else:
            bandwidth = cfg.B_C_TO_E
            power = cfg.p_C_TO_E
            channel_gain = cfg.h_C_TO_E

        signal_power = power * channel_gain
        rate = bandwidth * np.log2(1 + signal_power / cfg.N0) if cfg.N0 > 0 else 0
        latency = data_size * 8 / rate if rate > 0 else float('inf')
        energy = power * latency
        return latency, energy

    def _calculate_inference_metrics(self, model_type: str, input_length: int,
                                     output_length: int = None, avg_attention_weight: float = 0.0) -> Tuple[
        float, float]:
        if model_type == self.cloud_model:
            params = self.cloud_params
            n = output_length if output_length else cfg.MAX_NEW_TOKENS
        else:
            params = self.edge_params[model_type]
            n = output_length if output_length else cfg.MAX_NEW_TOKENS // 2

        base_latency = params["latency"] * (input_length / 1000) * (1 + n / 50)
        discount_factor = (1 - self.epsilon_discount * avg_attention_weight)
        discounted_latency = base_latency * discount_factor
        latency = base_latency * (1 + np.random.uniform(-0.1, 0.1))
        energy = params["power"] * latency
        return latency, energy

    def calculate_cloud_latency_theory(self, model_name: str, input_length: int,
                                       output_length: int, avg_attention_weight: float = 0.0) -> float:
        params = cfg.MODEL_PERFORMANCE[model_name]
        capacity = params["capacity"]
        k1 = params.get("k1", 0.0)
        k2 = params.get("k2", 0.0)
        k3 = params.get("k3", 0.0)
        n = output_length if output_length else cfg.MAX_NEW_TOKENS

        base_latency = (1 / capacity) * (k1 + k2 * n + k3 * (n ** 2))
        discount = self.epsilon_discount * avg_attention_weight
        inference_latency = max(base_latency - discount, 0.001)

        data_size = input_length * 4
        transmission_latency, _ = self._calculate_transmission_metrics(data_size, is_uplink=True)
        total_latency = transmission_latency + inference_latency
        return max(total_latency, 0.001)

    def calculate_edge_latency_theory(self, model_name: str, input_length: int,
                                      output_length: int, avg_attention_weight: float = 0.0) -> float:
        params = cfg.MODEL_PERFORMANCE[model_name]
        capacity = params["capacity"]
        k1 = params.get("k1", 0.0)
        k2 = params.get("k2", 0.0)
        k3 = params.get("k3", 0.0)
        n = output_length if output_length else cfg.MAX_NEW_TOKENS // 2

        base_latency = (1 / capacity) * (k1 + k2 * n + k3 * (n ** 2))
        discount = self.epsilon_discount * avg_attention_weight
        inference_latency = max(base_latency - discount, 0.001)

        data_size = input_length * 4
        transmission_latency, _ = self._calculate_transmission_metrics(data_size, is_uplink=True)
        total_latency = transmission_latency + inference_latency
        return max(total_latency, 0.001)


class SketchGenerator:
    def __init__(self):
        self.sketch_template = """Generate {count} diverse answer sketches for the query below, each under 50 words:

    Query: {query}

    Requirements:
    1. Mark each sketch with [Sketch X] (X=1,2,...)
    2. Each sketch should focus on a distinct reasoning direction
    3. Keep semantics concise and accurate

    Output format:
    [Sketch 1] [Content1]
    [Sketch 2] [Content2]
    ..."""

    def generate_sketches(self, query: str, history_context: str = None) -> List[Dict]:
        try:
            prompt = self.sketch_template.format(count=min(cfg.MAX_SKETCH_NUM, len(cfg.EDGE_SERVERS)), query=query)

            if history_context:
                history_guide = f"""【History Reference - FOR CONTEXT ONLY, DO NOT ANSWER THESE】:
            {history_context}

            Important Instruction: When generating sketches, ONLY use the history to understand the user's background. Focus EXCLUSIVELY on the "Current Query to Answer" — do NOT respond to any questions in the history.
            """
                prompt = history_guide + "\n\n" + prompt

            messages = [
                {"role": "system",
                 "content": "You are a professional reasoning sketch generator. Generate sketches ONLY for the 'Current Query to Answer' (ignore history)."},
                {"role": "user", "content": prompt}
            ]

            response = cloud_client.chat_completions(
                messages=messages,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_NEW_TOKENS
            )

            content = response["choices"][0].message.content if response["choices"] else ""
            sketches = self._parse_sketches(content)
            return sketches

        except Exception:
            return self._generate_default_sketches(query)

    def _parse_sketches(self, content: str) -> List[Dict]:
        sketches = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('[Sketch'):
                parts = line.split(']', 1)
                if len(parts) == 2:
                    sketch_id = parts[0].replace('[Sketch', '').strip()
                    sketch_content = parts[1].strip()
                    if sketch_id.isdigit() and 1 <= int(sketch_id) <= cfg.MAX_SKETCH_NUM:
                        sketches.append({
                            "id": int(sketch_id),
                            "content": sketch_content,
                            "token_length": calculate_token_length(sketch_content)
                        })
        return sketches[:len(cfg.EDGE_SERVERS)]

    def _generate_default_sketches(self, query: str) -> List[Dict]:
        return [
                   {"id": 1, "content": f"{query} Fact", "token_length": 20},
                   {"id": 2, "content": f"{query} Analysis", "token_length": 25},
                   {"id": 3, "content": f"{query} Create", "token_length": 22}
               ][:len(cfg.EDGE_SERVERS)]


class EdgeInferenceOrchestrator:
    def __init__(self):
        self.sketch_generator = SketchGenerator()
        self.calculator = PerformanceCalculator()
        self.inference_system = CloudEdgeInferenceSystem()

    def execute_cooperative_inference(self, query: str, history_context: str = None,
                                      reference_answer: str = None, sketches: List[Dict] = None,
                                      avg_attention_weight: float = 0.0) -> Dict:
        start_time = time.time()
        if sketches is None:
            sketches = self.sketch_generator.generate_sketches(query, history_context)
            input_length = calculate_token_length(query)
            if history_context:
                input_length += calculate_token_length(history_context)
            sketch_output_length = sum([s.get('token_length', 50) for s in sketches])

            theoretical_sketch_time = self.inference_system.calculate_cloud_latency_theory(
                cfg.CLOUD_MODEL,
                input_length,
                sketch_output_length,
                avg_attention_weight
            )
        else:
            theoretical_sketch_time = 0.0

        edge_results = self._distribute_to_edge_servers(sketches, query, avg_attention_weight)
        final_result = self._integrate_results(edge_results, query)
        final_response_cleaned = clean_final_response(final_result)

        total_latency, total_energy, accuracy = self.calculator.calculate_metrics_theory(
            theoretical_sketch_time,
            edge_results,
            final_response_cleaned,
            reference_answer,
            avg_attention_weight
        )

        return {
            "final_answer": final_response_cleaned,
            "sketches": sketches,
            "edge_results": edge_results,
            "metrics": {
                "total_latency": total_latency,
                "total_energy": total_energy,
                "accuracy": accuracy,
                "sketch_generation_time": theoretical_sketch_time
            }
        }

    def _distribute_to_edge_servers(self, sketches: List[Dict], query: str,
                                    avg_attention_weight: float = 0.0) -> List[InferenceResult]:
        results = []
        with ThreadPoolExecutor(max_workers=len(cfg.EDGE_SERVERS)) as executor:
            future_to_sketch = {}
            for i, sketch in enumerate(sketches):
                if i < len(cfg.EDGE_SERVERS):
                    server_id = cfg.EDGE_SERVERS[i]["id"]
                    future = executor.submit(
                        self._execute_edge_inference,
                        sketch, query, server_id, avg_attention_weight
                    )
                    future_to_sketch[future] = sketch

            for future in as_completed(future_to_sketch):
                sketch = future_to_sketch[future]
                try:
                    result = future.result()
                    result.sketch_id = sketch["id"]
                    results.append(result)
                except Exception as e:
                    results.append(InferenceResult(
                        content=f"Error: {str(e)}",
                        latency=0,
                        energy=0,
                        model_name="unknown",
                        sketch_id=sketch["id"]
                    ))
        return results

    def _execute_edge_inference(self, sketch: Dict, query: str, server_id: int,
                                avg_attention_weight: float = 0.0) -> InferenceResult:
        server = next((s for s in cfg.EDGE_SERVERS if s["id"] == server_id), None)
        if not server:
            raise ValueError(f"Edge server {server_id} does not exist")
        model_name = server["model"]

        prompt = f"""
        Generate a response based on the following sketch.

        Requirements:
        - Directly output the core content without any others.
        - Keep the language concise and focused on the topic.

        Sketch: [{sketch['content']}]
        Original query: {query}

        Your response:
        """

        messages = [{"role": "user", "content": prompt}]
        response = edge_clients[model_name].chat_completions(
            messages=messages,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_NEW_TOKENS // 2
        )

        content = response["choices"][0].message.content if response["choices"] else ""

        input_length = calculate_token_length(prompt)
        output_length = calculate_token_length(content)

        theoretical_latency = self.inference_system.calculate_edge_latency_theory(
            model_name,
            input_length,
            output_length,
            avg_attention_weight
        )

        energy = self.calculator._calculate_edge_energy_theory(
            model_name, theoretical_latency
        )

        return InferenceResult(
            content=content,
            latency=theoretical_latency,
            energy=energy,
            model_name=model_name
        )

    def _integrate_results(self, edge_results: List[InferenceResult], query: str) -> str:
        sorted_results = sorted(edge_results, key=lambda x: x.sketch_id if x.sketch_id else 0)
        integrated_content = f"Comprehensive analysis for: {query}\n\n"
        for i, result in enumerate(sorted_results):
            integrated_content += f"[Perspective {i + 1} - {result.model_name}]:\n"
            integrated_content += f"{result.content}\n\n"
        return integrated_content


class PerformanceCalculator:
    def calculate_metrics_theory(self, sketch_time: float,
                                 edge_results: List[InferenceResult],
                                 final_response_cleaned: str,
                                 reference_answer: str = None,
                                 avg_attention_weight: float = 0.0) -> Tuple[float, float, float]:
        edge_latency = max([result.latency for result in edge_results]) if edge_results else 0
        total_latency = sketch_time + edge_latency
        total_energy = self._calculate_cloud_energy(sketch_time) + \
                       sum(result.energy for result in edge_results)

        accuracy = 0.0
        if reference_answer:
            accuracy = calculate_bertscore(final_response_cleaned, reference_answer)

        return total_latency, total_energy, accuracy

    def _calculate_cloud_energy(self, latency: float) -> float:
        return cfg.CLOUD_POWER * latency

    def _calculate_edge_energy_theory(self, model_name: str, latency: float) -> float:
        params = cfg.MODEL_PERFORMANCE[model_name]
        power = params.get("power", 100)
        return power * latency


def run_cooperative_inference_experiment(query: str, history_context: str = None,
                                         reference_answer: str = None) -> Dict:
    system = CloudEdgeInferenceSystem()
    orchestrator = EdgeInferenceOrchestrator()
    calculator = PerformanceCalculator()

    result = orchestrator.execute_cooperative_inference(
        query, history_context, reference_answer
    )
    return result