import numpy as np
from gymnasium import spaces
from openenv.core import Environment

class IncidentCommanderEnv(Environment):
    def __init__(self):
        super().__init__()
        
        # 1. Action Space (The SRE Toolkit)
        # 0: Observe (Do nothing)
        # 1: Restart API Gateway
        # 2: Rollback Auth Service
        # 3: Scale Up Database
        # 4: Clear Redis Cache
        self.action_space = spaces.Discrete(5)
        
        # 2. Observation Space (Server Telemetry)
        # [API_CPU, API_Lat, Auth_CPU, Auth_Err, DB_CPU, DB_Lat, Global_SLA]
        self.observation_space = spaces.Box(
            low=0.0, high=10000.0, shape=(7,), dtype=np.float32
        )
        
        self.max_steps = 15
        self.task_name = "hard-cascading-failure"
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initial Healthy State
        self.state = {
            "api_cpu": 30.0, "api_lat": 45.0, 
            "auth_cpu": 25.0, "auth_err": 0.1,
            "db_cpu": 40.0, "db_lat": 10.0
        }
        
        self.global_sla = 100.0 # Starts at 100% Uptime
        self.incident_resolved = False
        self.downtime_penalty_accumulated = 0.0

        obs = self._get_obs()
        info = {
            "sla": self.global_sla, 
            "score": float(self.score()),         
            "success": bool(self.score() >= 0.5)  
        }
        return obs, info

    def _get_obs(self):
        return np.array([
            self.state["api_cpu"], self.state["api_lat"],
            self.state["auth_cpu"], self.state["auth_err"],
            self.state["db_cpu"], self.state["db_lat"],
            self.global_sla
        ], dtype=np.float32)
    
    def get_state(self):
        """Mandatory method required by the OpenEnv Environment base class."""
        return self._get_obs()

    def score(self) -> float:
        """Official Grader: Returns SLA survival score strictly between (0, 1)."""
        # TASK 1: EASY (Just survive)
        if self.task_name == "easy-api-compliance":
            return 0.99 

        # TASK 2: MEDIUM (Single failure)
        elif self.task_name == "medium-auth-spike":
            if self.global_sla < 90.0:
                return 0.01 # Failed SLA
            elif self.incident_resolved:
                return 0.99
            else:
                return min(0.99, max(0.01, self.global_sla / 100.0))

        # TASK 3: HARD (Cascading failure)
        else:
            if self.global_sla < 85.0:
                return 0.01 # System Crashed
            if self.state["db_cpu"] >= 99.0 or self.state["api_lat"] > 2000.0:
                return 0.10 # Left the system burning
            elif self.incident_resolved:
                return 0.99 # Perfect fix
            else:
                # Math: Map 85-100 SLA to a 0.2 - 0.8 score
                raw_score = ((self.global_sla - 85.0) / 15.0) * 0.6 + 0.2
                return min(0.99, max(0.01, float(raw_score)))

    def step(self, action):
        reward = 0.0
        
        # --- THE CHAOS ENGINE (Task-Based Incidents) ---
        if self.task_name == "medium-auth-spike" and self.current_step == 3:
            # Bad code push at step 3!
            self.state["auth_err"] = 85.0
            self.state["api_lat"] = 1500.0
            
        elif self.task_name == "hard-cascading-failure" and self.current_step == 4:
            # Huge traffic spike hits the DB
            self.state["db_cpu"] = 99.9
            self.state["db_lat"] = 5000.0
            self.state["api_lat"] = 3000.0 # Cascades to API

        # --- ACTION HANDLER ---
        if action == 1: # Restart API
            self.state["api_lat"] = 45.0
            self.state["api_cpu"] = 30.0
        elif action == 2: # Rollback Auth
            if self.state["auth_err"] > 10.0:
                self.state["auth_err"] = 0.1
                self.state["api_lat"] = 50.0 # Clears the queue
                self.incident_resolved = True
                reward += 5.0 # Positive reinforcement for the right fix
        elif action == 3: # Scale DB
            if self.state["db_cpu"] > 90.0:
                self.state["db_cpu"] = 40.0
                self.state["db_lat"] = 10.0
                self.incident_resolved = True
                reward += 5.0

        # --- SLA TRACKER & PHYSICS ---
        # If any system is burning, SLA drops linearly
        is_burning = self.state["auth_err"] > 10.0 or self.state["db_cpu"] > 90.0 or self.state["api_lat"] > 1000.0
        if is_burning:
            self.global_sla -= 2.5
            reward -= 1.0 # Bleed reward while broken

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "sla": self.global_sla, 
            "score": float(self.score()),         
            "success": bool(self.score() >= 0.5)
        }

        return self._get_obs(), float(reward), terminated, truncated, info