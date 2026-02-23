import threading
import time
import numpy as np
from config import cfg
import logging

logger = logging.getLogger(__name__)

class EdgeServer:
    def __init__(self, server_id, model_name, capacity):
        self.server_id = server_id
        self.model_name = model_name
        self.capacity = capacity
        self.current_load = 0.0
        self.status = "idle"
        self.lock = threading.Lock()

    def calculate_required_cap(self, text):
        tokens = len(text.split())
        return tokens * 1e6

    def can_handle(self, text):
        required = self.calculate_required_cap(text)
        return (self.current_load + required) <= self.capacity

    def acquire(self, text, timeout=5):
        start = time.perf_counter()
        if not self.lock.acquire(timeout=timeout):
            return False
        try:
            required = self.calculate_required_cap(text)
            if self.can_handle(text):
                self.status = "busy"
                self.current_load += required
                return True
            return False
        finally:
            self.lock.release()

    def release(self):
        try:
            with self.lock:
                if self.status == "idle" and self.current_load <= 0:
                    return
                self.status = "idle"
                self.current_load = 0.0
        except Exception as e:
            self.status = "idle"
            self.current_load = 0.0

class EdgeManager:
    def __init__(self):
        self.servers = {}
        for srv in cfg.EDGE_SERVERS:
            srv_id = srv["id"]
            model_name = srv["model"]
            if model_name not in cfg.EDGE_MODELS:
                continue
            self.servers[srv_id] = EdgeServer(
                server_id=srv_id,
                model_name=model_name,
                capacity=srv["capacity"]
            )
        if not self.servers:
            raise RuntimeError("Init error")
        self.lock = threading.Lock()

    def get_server_info(self, server_id):
        srv = self.servers.get(server_id)
        if not srv:
            return None
        return {
            "id": srv.server_id,
            "model": srv.model_name,
            "status": srv.status,
            "current_load": srv.current_load,
            "capacity": srv.capacity
        }

    def acquire_resource(self, server_id, text, timeout=5):
        srv = self.servers.get(server_id)
        if not srv:
            return False
        return srv.acquire(text, timeout=timeout)

    def release_resource(self, server_id):
        srv = self.servers.get(server_id)
        if srv:
            srv.release()

    def get_all_servers_status(self):
        with self.lock:
            return [
                {
                    "id": server.server_id,
                    "model": server.model_name,
                    "status": server.status,
                    "current_load": server.current_load,
                    "capacity": server.capacity
                }
                for server in self.servers.values()
            ]

    def get_average_resource_usage(self) -> float:
        servers_status = self.get_all_servers_status()
        utilizations = []
        for srv in servers_status:
            srv_obj = self.servers.get(srv["id"])
            if srv_obj and srv_obj.capacity > 0:
                utilization = srv_obj.current_load / srv_obj.capacity
                utilizations.append(utilization)
        return np.mean(utilizations) if utilizations else 0.0

edge_manager = EdgeManager()