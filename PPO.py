import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import deque
import random
import time
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from chroma_db import ChromaHistoryDB
from config import cfg
from cloud_edge_inference import CloudEdgeInferenceSystem, EdgeInferenceOrchestrator, SketchGenerator
from edge_resource_manager import edge_manager
from history_elc import attention_refinement
from init_LLM import edge_clients
from utils import calculate_bertscore, calculate_token_length
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class CriticalAPIError(Exception):
    pass


class TemporaryAPIError(Exception):
    pass


@dataclass
class PPOResult:
    user_id: str
    turn_number: int
    question: str
    K_i: int
    x_i: int
    y_gj: Dict[int, int]
    L_i: float
    performance_metrics: Dict[str, float]
    reward: float


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(PPONetwork, self).__init__()
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor_K = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims['K_i']),
            nn.Softmax(dim=-1)
        )
        self.actor_x = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_dims['x_i']),
            nn.Softmax(dim=-1)
        )
        self.actor_y = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims['y_gj']),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        shared_features = self.shared_network(state)
        prob_K = self.actor_K(shared_features)
        prob_x = self.actor_x(shared_features)
        prob_y = self.actor_y(shared_features)
        value = self.critic(shared_features)
        return prob_K, prob_x, prob_y, value


class PPOAgent:
    def __init__(self, state_dim, action_dims, chroma_dase):
        self._current_avg_weight = None
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.chroma_db = chroma_dase
        self.policy_net = PPONetwork(state_dim, action_dims)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4, eps=1e-5)

        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.epsilon = 0.15
        self.epochs = 6
        self.batch_size = 128
        self.buffer = deque(maxlen=10000)
        self.inference_system = CloudEdgeInferenceSystem()
        self.orchestrator = EdgeInferenceOrchestrator()
        self.sketchs = SketchGenerator()
        self.results_cache = {}
        self.epoch_decisions = []
        self.best_avg_reward = -float('inf')
        self.no_improvement_count = 0
        self.patience = 300
        self.training_history = []

        self.weights = {
            'utility': 2.0,
            'energy': 0.5,
            'deadline': 1.5,
            'accuracy': 1.2,
            'matching': 0.8
        }

        self.emergency_checkpoint_path = "emergency_checkpoint.pth"
        self.api_retry_count = 1
        self.api_retry_delay = 2
        self.current_processing_index = 0
        self.current_episode = 0
        self._checkpoint_saved = False

    def save_emergency_checkpoint(self, episode: int, data_index: int, error_info: str = None):
        if self._checkpoint_saved:
            return
        checkpoint = {
            'episode': episode,
            'data_index': data_index,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer': list(self.buffer),
            'training_history': self.training_history,
            'epoch_decisions': self.epoch_decisions,
            'network_architecture': {
                'state_dim': self.state_dim,
                'action_dims': self.action_dims,
            },
            'hyperparameters': {
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'epsilon': self.epsilon,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            },
            'training_state': {
                'best_avg_reward': self.best_avg_reward,
                'no_improvement_count': self.no_improvement_count,
                'weights': self.weights.copy()
            },
            'error_info': {
                'timestamp': time.time(),
                'error_message': str(error_info) if error_info else None,
                'episode': episode,
                'data_index': data_index
            },
            'results_cache': self.results_cache
        }
        torch.save(checkpoint, self.emergency_checkpoint_path)
        self._checkpoint_saved = True

    @classmethod
    def load_from_emergency_checkpoint(cls, checkpoint_path: str, chroma_db):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            agent = cls(
                state_dim=checkpoint['network_architecture']['state_dim'],
                action_dims=checkpoint['network_architecture']['action_dims'],
                chroma_dase=chroma_db
            )
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.gamma = checkpoint['hyperparameters']['gamma']
            agent.lambda_gae = checkpoint['hyperparameters']['lambda_gae']
            agent.epsilon = checkpoint['hyperparameters']['epsilon']
            agent.epochs = checkpoint['hyperparameters']['epochs']
            agent.batch_size = checkpoint['hyperparameters']['batch_size']
            agent.best_avg_reward = checkpoint['training_state']['best_avg_reward']
            agent.no_improvement_count = checkpoint['training_state']['no_improvement_count']
            agent.weights = checkpoint['training_state']['weights']
            agent.buffer = deque(checkpoint['buffer'], maxlen=10000)
            agent.training_history = checkpoint['training_history']
            agent.epoch_decisions = checkpoint['epoch_decisions']
            agent.results_cache = checkpoint.get('results_cache', {})
            agent.current_episode = checkpoint['episode']
            agent.current_processing_index = checkpoint['data_index']
            return agent, checkpoint['episode'], checkpoint['data_index']
        except Exception as e:
            raise

    def _api_call_with_retry(self, api_func, *args, **kwargs):
        last_error = None
        for attempt in range(self.api_retry_count):
            try:
                result = api_func(*args, **kwargs)
                if result is None:
                    raise ValueError("Error")
                if isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()):
                    raise ValueError(f"Error {result}")
                if isinstance(result, dict):
                    if "choices" in result and len(result["choices"]) == 0:
                        raise ValueError("Error")
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                is_resource_exhausted = any(keyword in error_msg for keyword in [
                    'quota', 'rate limit', 'too many requests', '429',
                    'insufficient', 'exceeded', 'token limit', 'billing',
                    'credit', 'usage limit', 'balance', 'setlimitexceeded',
                    'has been paused', 'safe experience mode'
                ])
                is_permanent = any(keyword in error_msg for keyword in [
                    '404', 'not found', 'unauthorized', '401', '403',
                    'forbidden', 'invalid api key', 'authentication failed'
                ])
                if is_resource_exhausted or is_permanent:
                    error_type = "Resource exhausted" if is_resource_exhausted else "Permanent Error"
                    raise CriticalAPIError(f"{error_type}: {str(e)}") from e
                if attempt < self.api_retry_count - 1:
                    time.sleep(self.api_retry_delay)
        error_msg = f"API error"
        raise TemporaryAPIError(error_msg) from last_error

    def save_complete_checkpoint(self, path: str, episode: int):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_architecture': {
                'state_dim': self.state_dim,
                'action_dims': self.action_dims,
                'hidden_layers': [256, 128]
            },
            'hyperparameters': {
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'epsilon': self.epsilon,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': 1e-4
            },
            'training_state': {
                'episode': episode,
                'best_avg_reward': self.best_avg_reward,
                'no_improvement_count': self.no_improvement_count,
                'training_progress': getattr(self, 'training_progress', 0.0),
                'weights': self.weights.copy()
            },
            'environment_config': {
                'edge_servers': cfg.EDGE_SERVERS,
                'cloud_model': cfg.CLOUD_MODEL
            },
            'buffer_sample': list(self.buffer)[-1000:] if self.buffer else []
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_complete_checkpoint(cls, checkpoint_path: str, chroma_db):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            agent = cls(
                state_dim=checkpoint['network_architecture']['state_dim'],
                action_dims=checkpoint['network_architecture']['action_dims'],
                chroma_dase=chroma_db
            )
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.gamma = checkpoint['hyperparameters']['gamma']
            agent.lambda_gae = checkpoint['hyperparameters']['lambda_gae']
            agent.epsilon = checkpoint['hyperparameters']['epsilon']
            agent.epochs = checkpoint['hyperparameters']['epochs']
            agent.batch_size = checkpoint['hyperparameters']['batch_size']
            agent.best_avg_reward = checkpoint['training_state']['best_avg_reward']
            agent.no_improvement_count = checkpoint['training_state']['no_improvement_count']
            agent.weights = checkpoint['training_state']['weights']
            if 'buffer_sample' in checkpoint:
                agent.buffer = deque(checkpoint['buffer_sample'], maxlen=10000)
            return agent
        except Exception as e:
            raise

    def calculate_K_max_optimized(self, C_ap: float, d_attn: int, d_ffn: int,
                                  L_q_i: int, historical_context: List[str]) -> int:
        if not historical_context:
            return 1
        from utils import calculate_token_length
        hist_lengths = [calculate_token_length(text) for text in historical_context]
        L_R_avg = np.mean(hist_lengths) if hist_lengths else 0
        if L_R_avg <= 0:
            return 1
        denominator = 4 * L_R_avg * d_attn * (d_attn + d_ffn)
        if denominator == 0:
            return 1
        K_max = int(np.floor(C_ap / denominator)) - L_q_i
        return max(K_max, 1)

    def get_state(self, user_data: Dict, historical_data: List[Dict]) -> Tuple[np.ndarray, int]:
        count = 0
        user_features = [
            user_data['tau_i'],
            user_data['a_i'],
            calculate_token_length(user_data['user_question']) / 100
        ]
        edge_resources = []
        for server_id in range(1, len(cfg.EDGE_SERVERS) + 1):
            server_info = edge_manager.get_server_info(server_id)
            if server_info:
                edge_resources.extend([
                    server_info['current_load'] / server_info['capacity'],
                    1.0 if server_info['status'] == 'idle' else 0.0
                ])

        has_historical_data = bool(historical_data) and len(historical_data) > 0
        if has_historical_data:
            matching_scores = [d['L_i'] for d in historical_data if 'L_i' in d]
            if matching_scores:
                historical_stats = [
                    np.mean(matching_scores),
                    np.std(matching_scores),
                    np.max(matching_scores),
                    np.min(matching_scores),
                    len(matching_scores) / 10
                ]
            else:
                historical_stats = [0.0] * 3
        else:
            historical_stats = [0.0] * 3

        ap_resources = [
            cfg.C_ap / 1e12,
            cfg.d_attn / 1000,
            cfg.d_ffn / 1000
        ]

        if has_historical_data:
            history_lengths = [calculate_token_length(d.get('history_text', ''))
                               for d in historical_data if isinstance(d, dict)]
            avg_history_length = np.mean(history_lengths) / 100 if history_lengths else 0.0
        else:
            avg_history_length = 0.0

        state_components = user_features + edge_resources + historical_stats + ap_resources + [avg_history_length]
        state = np.zeros(30)
        state[:len(state_components)] = state_components
        count += len(state_components)

        if len(state) < 30:
            state = np.pad(state, (0, 30 - len(state)))
        elif len(state) > 30:
            state = state[:30]

        return state, count

    def select_action_K(self, state, K_max):
        with torch.no_grad():
            prob_K, _, _, value = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            if hasattr(self, 'training_progress'):
                epsilon = max(0.1, 0.5 * (1 - self.training_progress))
                if np.random.random() < epsilon:
                    action = np.random.randint(0, K_max)
                    return action, prob_K[0][action].item(), value.item()
            mask = torch.zeros_like(prob_K)
            mask[:, :K_max] = 1.0
            masked_probs = prob_K * mask
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            dist = torch.distributions.Categorical(masked_probs)
            action = dist.sample()
            return action.item(), masked_probs[0][action].item(), value.item()

    def select_action_x(self, state):
        with torch.no_grad():
            _, prob_x, _, value = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            dist = torch.distributions.Categorical(prob_x)
            action = dist.sample()
            return action.item(), prob_x[0][action].item(), value.item()

    def select_action_y(self, state):
        with torch.no_grad():
            _, _, prob_y, value = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            dist = torch.distributions.Categorical(prob_y)
            action = dist.sample()
            return action.item(), prob_y[0][action].item(), value.item()

    def execute_action_K(self, user_data, historical_context, K_i):
        selected_history = historical_context[:K_i] if historical_context else []
        if selected_history:
            from history_elc import attention_refinement
            _, L_i, _, _, avg_weight = attention_refinement(
                user_data['user_question'],
                selected_history
            )
            if L_i is None or not (0 <= L_i <= 1):
                L_i = 0.0
        else:
            L_i = 0.0
            avg_weight = 0.0
        return L_i, selected_history, avg_weight

    def execute_action_x_safe(self, user_data, selected_history, x_i):
        if x_i == 1:
            def _generate_sketches():
                return self.sketchs.generate_sketches(user_data['user_question'])

            sketches = self._api_call_with_retry(_generate_sketches)
            if not sketches or len(sketches) == 0:
                raise ValueError("Error sketches")

            sketch_lengths = [sketch.get('token_length', 50) for sketch in sketches]
            input_length = calculate_token_length(user_data['user_question'])

            if selected_history:
                history_length = calculate_token_length(" ".join(selected_history))
                input_length += history_length

            sketch_output_length = sum(sketch_lengths)
            _, _, _, _, avg_weight = attention_refinement(
                user_data['user_question'],
                selected_history
            )

            theoretical_sketch_time = self.inference_system.calculate_cloud_latency_theory(
                cfg.CLOUD_MODEL,
                input_length,
                sketch_output_length,
                avg_weight
            )
            return sketch_lengths, sketches, theoretical_sketch_time
        else:
            task_length = calculate_token_length(user_data['user_question'])
            return [task_length], None, 0.0

    def execute_action_y_safe(self, user_data, selected_history, x_i, y_gj,
                              sketches=None, sketch_time=0.0, avg_attention_weight=0.0):
        if x_i == 0:
            context = " ".join(selected_history) if selected_history else ""
            full_query = f"{context} {user_data['user_question']}" if context else user_data['user_question']
            server_id = y_gj + 1
            server_info = edge_manager.get_server_info(server_id)
            model_name = server_info['model']
            result_text = self._execute_edge_inference_safe(full_query, y_gj)

            input_length = calculate_token_length(full_query)
            output_length = calculate_token_length(result_text)
            theoretical_latency = self.inference_system.calculate_edge_latency_theory(
                model_name, input_length, output_length, avg_attention_weight
            )
            theoretical_energy = cfg.MODEL_PERFORMANCE[model_name]["power"] * theoretical_latency
            inference_result = {
                'final_answer': result_text,
                'metrics': {
                    'total_energy': theoretical_energy
                }
            }
            total_time = theoretical_latency
        else:
            def _cooperative_inference():
                return self.orchestrator.execute_cooperative_inference(
                    query=user_data['user_question'],
                    history_context=" ".join(selected_history) if selected_history else "",
                    reference_answer=user_data.get('model_answer', ''),
                    sketches=sketches,
                    avg_attention_weight=avg_attention_weight
                )

            inference_result = self._api_call_with_retry(_cooperative_inference)
            total_time = inference_result['metrics']['total_latency']

        accuracy = calculate_bertscore(inference_result['final_answer'],
                                       user_data['model_answer'])
        performance_metrics = {
            'total_time': total_time,
            'sketch_time': sketch_time if x_i == 1 else 0.0,
            'inference_time': total_time - sketch_time if x_i == 1 else total_time,
            'accuracy': accuracy,
            'energy_consumption': inference_result['metrics']['total_energy'],
            'meets_deadline': 1.0 if total_time <= user_data['tau_i'] else 0.0,
            'meets_accuracy': 1.0 if accuracy >= user_data['a_i'] else 0.0
        }
        return inference_result, performance_metrics, sketch_time

    def train_with_checkpoint_recovery(self, dataset: List[Dict], num_episodes: int,
                                       checkpoint_path: str = "ppo_checkpoint.pth",
                                       resume_from_emergency: bool = False):
        start_episode = 0
        start_data_index = 0

        if resume_from_emergency and os.path.exists(self.emergency_checkpoint_path):
            try:
                checkpoint = torch.load(self.emergency_checkpoint_path, weights_only=False)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.buffer = deque(checkpoint['buffer'], maxlen=10000)
                self.training_history = checkpoint['training_history']
                self.epoch_decisions = checkpoint['epoch_decisions']
                self.results_cache = checkpoint.get('results_cache', {})
                self.best_avg_reward = checkpoint['training_state']['best_avg_reward']
                self.no_improvement_count = checkpoint['training_state']['no_improvement_count']
                self.weights = checkpoint['training_state']['weights']
                start_episode = checkpoint['episode']
                start_data_index = checkpoint['data_index'] + 1
            except Exception as e:
                start_episode = 0
                start_data_index = 0

        elif os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint['episode'] + 1
                self.buffer = checkpoint['buffer']
                self.training_history = checkpoint['training_history']
                self.epoch_decisions = checkpoint['epoch_decisions']
            except Exception as e:
                pass

        historical_results = []
        recent_rewards = deque(maxlen=5)
        recent_times = deque(maxlen=5)
        recent_accuracies = deque(maxlen=5)

        for episode in range(start_episode, num_episodes):
            self.current_episode = episode
            episode_rewards = []
            data_start_index = start_data_index if episode == start_episode else 0

            for i in range(data_start_index, len(dataset)):
                self.current_processing_index = i
                user_data = dataset[i]

                try:
                    state, count = self.get_state(user_data, historical_results[-10:])

                    actions, action_probs, value, exec_result, reward, updated_state = (
                        self.sequential_decision_and_execution(
                            state, count, user_data, episode, episode_rewards
                        )
                    )

                    episode_rewards.append(reward)

                    next_user_data = dataset[(i + 1) % len(dataset)]
                    next_state, _ = self.get_state(next_user_data, historical_results[-10:])

                    self.store_experience(
                        updated_state, actions, action_probs, value, reward,
                        next_state, i == len(dataset) - 1
                    )

                    try:
                        history_text = (f"Turn {user_data['turn_number']}: "
                                        f"User asked: {user_data['user_question']} "
                                        f"→ Assistant answered:{user_data['model_answer']}")
                        new_history = {user_data['user_id']: [(user_data['turn_number'], history_text)]}
                        self.chroma_db.add_history_to_db(new_history)
                    except Exception as e:
                        pass

                    if (i + 1) % 10 == 0:
                        checkpoint = {
                            'episode': episode,
                            'data_index': i,
                            'policy_state_dict': self.policy_net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'buffer': self.buffer,
                            'training_history': self.training_history,
                            'epoch_decisions': self.epoch_decisions,
                            'results_cache': self.results_cache
                        }
                        torch.save(checkpoint, "auto_checkpoint.pth")

                except CriticalAPIError as e:
                    self.save_emergency_checkpoint(episode, i, str(e))
                    sys.exit(1)
                except TemporaryAPIError as e:
                    continue
                except Exception as e:
                    continue

            self.update()
            start_data_index = 0

            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_accuracy = np.mean([d['performance_metrics']['accuracy']
                                    for d in self.epoch_decisions
                                    if 'performance_metrics' in d]) if self.epoch_decisions else 0
            avg_time = np.mean([d['performance_metrics']['total_time']
                                for d in self.epoch_decisions
                                if 'performance_metrics' in d]) if self.epoch_decisions else 0

            recent_rewards.append(avg_reward)
            recent_times.append(avg_time)
            recent_accuracies.append(avg_accuracy)

            self.training_history.append({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'avg_accuracy': avg_accuracy,
                'avg_time': avg_time,
                'avg_energy_consumption': np.mean([d['performance_metrics']['energy_consumption']
                                                   for d in self.epoch_decisions]) if self.epoch_decisions else 0,
            })

            if (episode + 1) % 10 == 0:
                self.save_decisions(f"decisions_episode_{episode + 1}.json")
                self.save_training_history(f"training_history_episode_{episode + 1}.json")
                self.save_complete_checkpoint(f"complete_checkpoint_ep_{episode + 1}.pth", episode)

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                break

            checkpoint = {
                'episode': episode,
                'policy_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'buffer': self.buffer,
                'training_history': self.training_history,
                'epoch_decisions': self.epoch_decisions
            }
            torch.save(checkpoint, checkpoint_path)

        if os.path.exists(self.emergency_checkpoint_path):
            os.remove(self.emergency_checkpoint_path)

        return historical_results

    def should_use_cache(self, cache_key, decision_key, current_L_i, threshold=0.2):
        if cache_key not in self.results_cache or decision_key not in self.results_cache[cache_key]:
            return False
        cached_data = self.results_cache[cache_key][decision_key]
        cached_L_i = cached_data.get('L_i', 0)
        max_possible_diff = 1.0
        actual_diff = abs(current_L_i - cached_L_i)
        normalized_diff = actual_diff / max_possible_diff
        if normalized_diff > threshold:
            return False
        return True

    def sequential_decision_and_execution(self, state, count, user_data, episode, episode_rewards):
        historical_context = self._retrieve_historical_context(
            user_data['user_id'],
            user_data['turn_number'],
            user_data['user_question'],
            cfg.TOP_K
        )

        L_q_i = calculate_token_length(user_data['user_question'])
        K_max = self.calculate_K_max_optimized(
            C_ap=cfg.C_ap,
            d_attn=cfg.d_attn,
            d_ffn=cfg.d_ffn,
            L_q_i=L_q_i,
            historical_context=historical_context
        )

        action_K, prob_K, value_K = self.select_action_K(state, K_max)
        K_i = min(action_K + 1, K_max)

        L_i, selected_history, avg_attention_weight = self.execute_action_K(
            user_data, historical_context, K_i
        )
        self._current_avg_weight = avg_attention_weight

        state_x = state.copy()
        state_x[count] = L_i
        count += 1
        action_x, prob_x, value_x = self.select_action_x(state_x)
        x_i = action_x

        cache_key_x = (user_data['user_id'], user_data['turn_number'])
        decision_key_x = (x_i,)

        use_x_cache = self.should_use_cache(cache_key_x, decision_key_x, L_i, threshold=0.2)

        if use_x_cache and cache_key_x in self.results_cache and decision_key_x in self.results_cache[cache_key_x]:
            cached_x_result = self.results_cache[cache_key_x][decision_key_x]['result']
            lengths = cached_x_result['lengths']
            sketches = cached_x_result['sketches']
            sketch_time = cached_x_result['sketch_time']
            is_x_cached = True
        else:
            lengths, sketches, sketch_time = self.execute_action_x_safe(user_data, selected_history, x_i)
            is_x_cached = False
            if cache_key_x not in self.results_cache:
                self.results_cache[cache_key_x] = {}
            self.results_cache[cache_key_x][decision_key_x] = {
                'result': {
                    'lengths': lengths,
                    'sketches': sketches,
                    'sketch_time': sketch_time
                },
                'L_i': L_i
            }

        state_y = state_x.copy()
        if lengths is not None:
            if count < 30:
                state_y[count] = len(lengths)
                count += 1
            for i, length in enumerate(lengths):
                if count < 30:
                    state_y[count] = length
                    count += 1
                else:
                    break

        action_y, prob_y, value_y = self.select_action_y(state_y)
        y_gj = action_y

        cache_key = (user_data['user_id'], user_data['turn_number'])
        decision_key = (x_i, y_gj)

        use_y_cache = self.should_use_cache(cache_key, decision_key, L_i, threshold=0.2)

        if use_y_cache and cache_key in self.results_cache and decision_key in self.results_cache[cache_key]:
            cached_result = self.results_cache[cache_key][decision_key]['result']
            final_answer = cached_result['final_answer']
            performance_metrics = cached_result['performance_metrics']
            is_y_cached = True
        else:
            result, performance_metrics, actual_sketch_time = self.execute_action_y_safe(
                user_data, selected_history, x_i, y_gj, sketches, sketch_time,
                avg_attention_weight
            )
            final_answer = result['final_answer']
            is_y_cached = False
            if cache_key not in self.results_cache:
                self.results_cache[cache_key] = {}
            self.results_cache[cache_key][decision_key] = {
                'result': {
                    'final_answer': final_answer,
                    'performance_metrics': {
                        'total_time': performance_metrics.get('total_time', 0),
                        'sketch_time': performance_metrics.get('sketch_time', 0),
                        'inference_time': performance_metrics.get('inference_time', 0),
                        'accuracy': performance_metrics.get('accuracy', 0),
                        'energy_consumption': performance_metrics.get('energy_consumption', 0),
                        'meets_deadline': performance_metrics.get('meets_deadline', 0),
                        'meets_accuracy': performance_metrics.get('meets_accuracy', 0)
                    }
                },
                'L_i': L_i
            }

        reward = self.calculate_reward(user_data, performance_metrics, L_i, episode_rewards)

        decision_record = {
            'epoch': episode,
            'user_id': user_data['user_id'],
            'turn_number': user_data['turn_number'],
            'K_i': K_i,
            'x_i': x_i,
            'y_gj': y_gj,
            'L_i': L_i,
            'performance_metrics': performance_metrics,
            'final_answer': final_answer,
            'reward': reward,
            'is_x_cached': is_x_cached,
            'is_y_cached': is_y_cached
        }
        self.epoch_decisions.append(decision_record)

        actions = {'K_i': K_i, 'x_i': x_i, 'y_gj': y_gj}
        action_probs = {'K_i': prob_K, 'x_i': prob_x, 'y_gj': prob_y}
        value = value_y

        return actions, action_probs, value, {
            'L_i': L_i,
            'performance_metrics': performance_metrics,
            'final_answer': final_answer,
            'sketch_time': sketch_time
        }, reward, state_y

    def _execute_edge_inference_safe(self, query, server_index):
        def _call_api():
            server_id = server_index + 1
            server_info = edge_manager.get_server_info(server_id)
            if not server_info:
                raise ValueError(f"Error {server_id}")
            model_name = server_info['model']
            response = edge_clients[model_name].chat_completions(
                messages=[{"role": "user", "content": query}],
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_NEW_TOKENS
            )
            if not response or "choices" not in response:
                raise ValueError("Error API")
            content = response["choices"][0].message.content if response["choices"] else ""
            if not content:
                raise ValueError("Error Content")
            return content

        return self._api_call_with_retry(_call_api)

    def _calculate_edge_energy(self, server_index, start_time, input_length, output_length):
        try:
            server_id = server_index + 1
            server_info = edge_manager.get_server_info(server_id)
            if not server_info:
                return 0.0
            model_name = server_info['model']
            signal_power = cfg.p_U * cfg.h_U
            transmission_rate = cfg.B_U * np.log2(1 + signal_power / cfg.N0) if cfg.N0 > 0 else 0
            if transmission_rate == 0:
                transmission_energy = 0
            else:
                transmission_time = (input_length * 8) / transmission_rate
                transmission_energy = cfg.p_U * transmission_time
            model_params = cfg.MODEL_PERFORMANCE.get(model_name, {})
            if not model_params:
                model_power = 100
                latency_factor = 0.5
            else:
                model_power = model_params.get("power", 100)
                latency_factor = model_params.get("latency", 0.5)
            inference_time = latency_factor * (1 + input_length / 500) * (1 + output_length / 100)
            computation_energy = model_power * inference_time
            total_energy = transmission_energy + computation_energy
            return total_energy
        except Exception as e:
            return 0.0

    def calculate_reward(self, user_data, performance_metrics, L_i, episode_rewards):
        T_ratio = performance_metrics['total_time'] / user_data['tau_i']
        accuracy = performance_metrics['accuracy']
        current_energy = performance_metrics['energy_consumption']
        if not hasattr(self, 'energy_history'):
            self.energy_history = deque(maxlen=100)
        self.energy_history.append(current_energy)
        avg_energy = np.mean(self.energy_history) if self.energy_history else current_energy
        if hasattr(self, 'training_progress'):
            progress = self.training_progress
            if progress <= 0.25:
                alpha = progress / 0.25
                weights = {
                    'utility': 1.5 + 0.5 * alpha,
                    'energy': 0.4 + 0.2 * alpha,
                    'accuracy': 2.0 - 0.0 * alpha,
                    'deadline': 1.8 - 0.3 * alpha,
                    'matching': 0.6 + 0.3 * alpha
                }
            else:
                alpha = (progress - 0.25) / 0.25
                weights = {
                    'utility': 2.0,
                    'energy': 0.6 + 0.4 * alpha,
                    'accuracy': 2.0 - 0.5 * alpha,
                    'deadline': 1.5 - 0.3 * alpha,
                    'matching': 0.9 + 0.4 * alpha
                }
        else:
            weights = {
                'utility': 1.5,
                'energy': 0.4,
                'accuracy': 2.0,
                'deadline': 1.8,
                'matching': 0.6
            }
        base_reward = 5.0
        if T_ratio <= 1.0:
            time_bonus = 3.0 * (1 - T_ratio) ** 0.8
            if T_ratio <= 0.7:
                time_bonus += 2.0
            elif T_ratio <= 0.85:
                time_bonus += 1.0
        else:
            overtime_ratio = T_ratio - 1.0
            if overtime_ratio <= 0.1:
                time_bonus = -1.0 * overtime_ratio * 10
            elif overtime_ratio <= 0.3:
                time_bonus = -1.0 - 2.0 * (overtime_ratio - 0.1)
            else:
                time_bonus = -1.4 - 1.5 * (overtime_ratio - 0.3)
        accuracy_target = user_data['a_i']
        accuracy_diff = accuracy - accuracy_target
        if accuracy_diff >= 0:
            accuracy_bonus = 4.0 + 3.0 * (accuracy_diff ** 0.6)
            if accuracy >= 0.85:
                accuracy_bonus += 2.0
        else:
            deficit_ratio = abs(accuracy_diff) / accuracy_target
            if deficit_ratio <= 0.05:
                accuracy_bonus = 2.0 - 8.0 * deficit_ratio
            elif deficit_ratio <= 0.15:
                accuracy_bonus = -0.4 - 10.0 * (deficit_ratio - 0.05)
            else:
                accuracy_bonus = -1.4 - 5.0 * (deficit_ratio - 0.15)
        if avg_energy > 0:
            energy_ratio = current_energy / avg_energy
            if energy_ratio < 1.0:
                energy_bonus = 1.5 * (1.0 - energy_ratio)
            else:
                energy_bonus = -0.8 * (energy_ratio - 1.0) ** 0.7
        else:
            energy_bonus = 0.0
        if L_i > 0.3:
            matching_bonus = 1.2 * (L_i ** 0.8)
        else:
            matching_bonus = 0.0
        total_reward = (
                base_reward +
                weights['utility'] * time_bonus +
                weights['accuracy'] * accuracy_bonus +
                weights['energy'] * energy_bonus +
                weights['matching'] * matching_bonus
        )
        if hasattr(self, 'training_progress'):
            progress = self.training_progress
            if progress < 0.3:
                total_reward *= 1.15
            elif progress > 0.7:
                if T_ratio < 0.8 and accuracy > accuracy_target:
                    total_reward += 1.5
        if episode_rewards and len(episode_rewards) >= 10:
            reward_mean = np.mean(episode_rewards[-10:])
            reward_std = np.std(episode_rewards[-10:])
            lower_bound = reward_mean - 3.0 * reward_std
            upper_bound = reward_mean + 3.0 * reward_std
            total_reward = np.clip(total_reward, lower_bound, upper_bound)
        return float(total_reward)

    def store_experience(self, state, actions, action_probs, value, reward, next_state, done):
        if not isinstance(next_state, np.ndarray) or next_state.shape != (self.state_dim,):
            sys.exit(1)
        experience = {
            'state': state,
            'actions': actions,
            'action_probs': action_probs,
            'value': value,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.buffer.append(experience)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = {
            'K_i': torch.LongTensor([exp['actions']['K_i'] - 1 for exp in batch]),
            'x_i': torch.LongTensor([exp['actions']['x_i'] for exp in batch]),
            'y_gj': torch.LongTensor([exp['actions']['y_gj'] for exp in batch])
        }
        old_probs = {
            'K_i': torch.FloatTensor([exp['action_probs']['K_i'] for exp in batch]),
            'x_i': torch.FloatTensor([exp['action_probs']['x_i'] for exp in batch]),
            'y_gj': torch.FloatTensor([exp['action_probs']['y_gj'] for exp in batch])
        }
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.FloatTensor([exp['done'] for exp in batch])

        with torch.no_grad():
            _, _, _, next_values = self.policy_net(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - torch.FloatTensor([exp['value'] for exp in batch])

        for _ in range(self.epochs):
            new_probs_K, new_probs_x, new_probs_y, values = self.policy_net(states)

            dist_K = torch.distributions.Categorical(new_probs_K)
            dist_x = torch.distributions.Categorical(new_probs_x)
            dist_y = torch.distributions.Categorical(new_probs_y)

            new_log_probs_K = dist_K.log_prob(actions['K_i'])
            new_log_probs_x = dist_x.log_prob(actions['x_i'])
            new_log_probs_y = dist_y.log_prob(actions['y_gj'])

            old_log_probs_K = torch.log(old_probs['K_i'] + 1e-10)
            old_log_probs_x = torch.log(old_probs['x_i'] + 1e-10)
            old_log_probs_y = torch.log(old_probs['y_gj'] + 1e-10)

            ratios_K = torch.exp(new_log_probs_K - old_log_probs_K)
            ratios_x = torch.exp(new_log_probs_x - old_log_probs_x)
            ratios_y = torch.exp(new_log_probs_y - old_log_probs_y)

            surr1_K = ratios_K * advantages
            surr2_K = torch.clamp(ratios_K, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss_K = -torch.min(surr1_K, surr2_K).mean()

            surr1_x = ratios_x * advantages
            surr2_x = torch.clamp(ratios_x, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss_x = -torch.min(surr1_x, surr2_x).mean()

            surr1_y = ratios_y * advantages
            surr2_y = torch.clamp(ratios_y, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss_y = -torch.min(surr1_y, surr2_y).mean()

            critic_loss = nn.MSELoss()(values.squeeze(), targets)
            total_loss = actor_loss_K + actor_loss_x + actor_loss_y + 0.5 * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

    def save_training_history(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

    def _retrieve_historical_context(self, user_id: str, turn_number: int, current_question: str, K_i: int) -> List[
        str]:
        try:
            retrieval_result = self.chroma_db.ann_retrieval_with_time_discount(
                query=current_question,
                current_turn=turn_number,
                user_id=user_id,
                top_k=K_i,
                similarity_threshold=cfg.SIMILARITY_THRESHOLD
            )
            if retrieval_result["has_history"]:
                return retrieval_result["top_k_histories"]
            else:
                return []
        except Exception as e:
            return []

    def save_model(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_decisions(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_decisions, f, ensure_ascii=False, indent=2)


def run_ppo_experiment():
    tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
    embedding_model = AutoModel.from_pretrained(cfg.EMBEDDING_MODEL).to(cfg.DEVICE)
    chroma_db = ChromaHistoryDB(tokenizer, embedding_model)
    dataset = load_and_prepare_data("data/enriched_test_queries.parquet", sample_size=150)
    state_dim = 30
    action_dims = {
        'K_i': cfg.TOP_K,
        'x_i': 2,
        'y_gj': len(cfg.EDGE_SERVERS)
    }
    ppo_agent = PPOAgent(state_dim, action_dims, chroma_db)
    results = ppo_agent.train_with_checkpoint_recovery(
        dataset,
        num_episodes=300,
        checkpoint_path="ppo_checkpoint.pth",
        resume_from_emergency=False
    )
    analyze_results(results)
    save_results_to_file(results, "ppo_experiment_results.json")


def analyze_results(results: List[PPOResult]):
    total_queries = len(results)
    avg_reward = np.mean([r.reward for r in results]) if results else 0
    avg_accuracy = np.mean([r.performance_metrics.get('accuracy', 0) for r in results]) if results else 0
    avg_time = np.mean([r.performance_metrics.get('total_time', 0) for r in results]) if results else 0

    K_i_dist = [r.K_i for r in results]
    x_i_dist = [r.x_i for r in results]

    deadline_meet_rates = []
    accuracy_meet_rates = []
    energy_consumptions = []

    for r in results:
        deadline_meet = float(r.performance_metrics.get('meets_deadline', 0))
        accuracy_meet = float(r.performance_metrics.get('meets_accuracy', 0))
        energy = float(r.performance_metrics.get('energy_consumption', 0))

        deadline_meet_rates.append(deadline_meet)
        accuracy_meet_rates.append(accuracy_meet)
        energy_consumptions.append(energy)

    deadline_meet_rate = np.mean(deadline_meet_rates) if deadline_meet_rates else 0
    accuracy_meet_rate = np.mean(accuracy_meet_rates) if accuracy_meet_rates else 0
    avg_energy = np.mean(energy_consumptions) if energy_consumptions else 0


def load_and_prepare_data(file_path: str, sample_size: int = 100) -> List[Dict]:
    try:
        df = pd.read_parquet(file_path)
        if len(df) > sample_size:
            df = df.head(sample_size)

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
    except Exception as e:
        raise


def save_results_to_file(results: List[PPOResult], filename: str):
    import json
    try:
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types([r.__dict__ for r in results]),
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass


def run_trained_ppo_inference(checkpoint_path: str, test_dataset: List[Dict]):
    tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
    embedding_model = AutoModel.from_pretrained(cfg.EMBEDDING_MODEL).to(cfg.DEVICE)
    chroma_db = ChromaHistoryDB(tokenizer, embedding_model)

    try:
        ppo_agent = PPOAgent.load_complete_checkpoint(checkpoint_path, chroma_db)
        ppo_agent.policy_net.eval()
    except Exception as e:
        return

    inference_results = []

    for i, user_data in enumerate(test_dataset):
        try:
            state, count = ppo_agent.get_state(user_data, [])

            with torch.no_grad():
                actions, action_probs, value, exec_result, reward, updated_state = (
                    ppo_agent.sequential_decision_and_execution(
                        state, count, user_data, 0, []
                    )
                )

            result = PPOResult(
                user_id=user_data["user_id"],
                turn_number=user_data["turn_number"],
                question=user_data["user_question"],
                K_i=actions['K_i'],
                x_i=actions['x_i'],
                y_gj=actions['y_gj'],
                L_i=exec_result['L_i'],
                performance_metrics=exec_result['performance_metrics'],
                reward=reward
            )

            inference_results.append(result)
        except Exception as e:
            continue

    analyze_inference_results(inference_results)
    save_inference_results(inference_results, "ppo_trained_inference_results.json")

    return inference_results


def analyze_inference_results(results: List[PPOResult]):
    if not results:
        return

    total_queries = len(results)
    avg_reward = np.mean([r.reward for r in results]) if results else 0
    avg_accuracy = np.mean([r.performance_metrics.get('accuracy', 0) for r in results]) if results else 0
    avg_time = np.mean([r.performance_metrics.get('total_time', 0) for r in results]) if results else 0

    energy_consumptions = [r.performance_metrics.get('energy_consumption', 0) for r in results]
    avg_energy = np.mean(energy_consumptions) if energy_consumptions else 0

    deadline_meet_rates = []
    accuracy_meet_rates = []

    for r in results:
        deadline_meet = float(r.performance_metrics.get('meets_deadline', 0))
        accuracy_meet = float(r.performance_metrics.get('meets_accuracy', 0))

        deadline_meet_rates.append(deadline_meet)
        accuracy_meet_rates.append(accuracy_meet)

    deadline_meet_rate = np.mean(deadline_meet_rates) if deadline_meet_rates else 0
    accuracy_meet_rate = np.mean(accuracy_meet_rates) if accuracy_meet_rates else 0


def save_inference_results(results: List[PPOResult], filename: str):
    import json
    from datetime import datetime

    if not results:
        return

    total_queries = len(results)
    avg_reward = np.mean([r.reward for r in results]) if results else 0
    avg_accuracy = np.mean([r.performance_metrics.get('accuracy', 0) for r in results]) if results else 0
    avg_time = np.mean([r.performance_metrics.get('total_time', 0) for r in results]) if results else 0

    energy_consumptions = [r.performance_metrics.get('energy_consumption', 0) for r in results]
    avg_energy = np.mean(energy_consumptions) if energy_consumptions else 0

    deadline_meet_rates = []
    accuracy_meet_rates = []

    for r in results:
        deadline_meet = float(r.performance_metrics.get('meets_deadline', 0))
        accuracy_meet = float(r.performance_metrics.get('meets_accuracy', 0))

        deadline_meet_rates.append(deadline_meet)
        accuracy_meet_rates.append(accuracy_meet)

    deadline_meet_rate = np.mean(deadline_meet_rates) if deadline_meet_rates else 0
    accuracy_meet_rate = np.mean(accuracy_meet_rates) if accuracy_meet_rates else 0

    output = {
        'timestamp': datetime.now().isoformat(),
        'results': [r.__dict__ for r in results],
        'summary': {
            'total_queries': total_queries,
            'avg_reward': avg_reward,
            'avg_accuracy': avg_accuracy,
            'avg_processing_time': avg_time,
            'avg_energy_consumption': avg_energy,
            'deadline_meet_rate': deadline_meet_rate,
            'accuracy_meet_rate': accuracy_meet_rate
        }
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def run_ppo_experiment_with_recovery():
    results = None
    tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
    embedding_model = AutoModel.from_pretrained(cfg.EMBEDDING_MODEL).to(cfg.DEVICE)
    chroma_db = ChromaHistoryDB(tokenizer, embedding_model)

    dataset = load_and_prepare_data("data/enriched_test_queries.parquet", sample_size=50)

    state_dim = 30
    action_dims = {
        'K_i': cfg.TOP_K,
        'x_i': 2,
        'y_gj': len(cfg.EDGE_SERVERS)
    }

    resume_from_emergency = os.path.exists("emergency_checkpoint.pth")

    if resume_from_emergency:
        ppo_agent, start_episode, start_index = PPOAgent.load_from_emergency_checkpoint(
            "emergency_checkpoint.pth", chroma_db
        )
    else:
        ppo_agent = PPOAgent(state_dim, action_dims, chroma_db)

    try:
        results = ppo_agent.train_with_checkpoint_recovery(
            dataset,
            num_episodes=300,
            checkpoint_path="ppo_checkpoint.pth",
            resume_from_emergency=resume_from_emergency
        )

    except KeyboardInterrupt:
        ppo_agent.save_emergency_checkpoint(
            ppo_agent.current_episode,
            ppo_agent.current_processing_index,
            "User Interruption"
        )
    except Exception as e:
        pass

    return results


if __name__ == "__main__":
    run_ppo_experiment()