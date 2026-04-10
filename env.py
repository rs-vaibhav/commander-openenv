from openenv.core import Environment
from models import SREObservation, SREAction, SREState

class IncidentCommanderEnv(Environment[SREAction, SREObservation, SREState]):
    def __init__(self):
        super().__init__()
        
        self.max_steps = 15
        self.task_name = "hard-cascading-failure"
        self._state_dict = {}
        self.current_step = 0
        self.global_sla = 100.0
        self.incident_resolved = False
        self.downtime_penalty_accumulated = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initial Healthy State
        self._state_dict = {
            "api_cpu": 30.0, "api_lat": 45.0, 
            "auth_cpu": 25.0, "auth_err": 0.1,
            "db_cpu": 40.0, "db_lat": 10.0
        }
        
        self.global_sla = 100.0 # Starts at 100% Uptime
        self.incident_resolved = False
        self.downtime_penalty_accumulated = 0.0

        if options and "task_name" in options:
            self.task_name = options["task_name"]

        obs = self._get_obs()
        info = {
            "sla": self.global_sla, 
            "score": float(self.score()),         
            "success": bool(self.score() >= 0.5)  
        }
        return obs, info

    def _get_obs(self) -> SREObservation:
        return SREObservation(
            api_cpu=self._state_dict["api_cpu"],
            api_lat=self._state_dict["api_lat"],
            auth_cpu=self._state_dict["auth_cpu"],
            auth_err=self._state_dict["auth_err"],
            db_cpu=self._state_dict["db_cpu"],
            db_lat=self._state_dict["db_lat"],
            global_sla=self.global_sla
        )
    
    def state(self) -> SREState:
        """Mandatory method required by the OpenEnv Environment base class."""
        return SREState(
            observation=self._get_obs(),
            current_step=self.current_step,
            incident_resolved=self.incident_resolved,
            task_name=self.task_name,
            downtime_penalty_accumulated=self.downtime_penalty_accumulated
        )

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
            if self._state_dict["db_cpu"] >= 99.0 or self._state_dict["api_lat"] > 2000.0:
                return 0.10 # Left the system burning
            elif self.incident_resolved:
                return 0.99 # Perfect fix
            else:
                # Math: Map 85-100 SLA to a 0.2 - 0.8 score
                raw_score = ((self.global_sla - 85.0) / 15.0) * 0.6 + 0.2
                return min(0.99, max(0.01, float(raw_score)))

    def step(self, action_request: SREAction):
        action = action_request.action
        reward = 0.0
        
        # --- THE CHAOS ENGINE (Task-Based Incidents) ---
        if self.task_name == "medium-auth-spike" and self.current_step == 3:
            # Bad code push at step 3!
            self._state_dict["auth_err"] = 85.0
            self._state_dict["api_lat"] = 1500.0
            
        elif self.task_name == "hard-cascading-failure" and self.current_step == 4:
            # Huge traffic spike hits the DB
            self._state_dict["db_cpu"] = 99.9
            self._state_dict["db_lat"] = 5000.0
            self._state_dict["api_lat"] = 3000.0 # Cascades to API

        # --- ACTION HANDLER ---
        if action == 1: # Restart API
            self._state_dict["api_lat"] = 45.0
            self._state_dict["api_cpu"] = 30.0
        elif action == 2: # Rollback Auth
            if self._state_dict["auth_err"] > 10.0:
                self._state_dict["auth_err"] = 0.1
                self._state_dict["api_lat"] = 50.0 # Clears the queue
                self.incident_resolved = True
                reward += 5.0 # Positive reinforcement for the right fix
        elif action == 3: # Scale DB
            if self._state_dict["db_cpu"] > 90.0:
                self._state_dict["db_cpu"] = 40.0
                self._state_dict["db_lat"] = 10.0
                self.incident_resolved = True
                reward += 5.0

        # --- SLA TRACKER & PHYSICS ---
        # If any system is burning, SLA drops linearly
        is_burning = self._state_dict["auth_err"] > 10.0 or self._state_dict["db_cpu"] > 90.0 or self._state_dict["api_lat"] > 1000.0
        if is_burning:
            self.global_sla -= 2.5
            reward -= 1.0 # Bleed reward while broken

        self.current_step += 1
        done = bool(self.current_step >= self.max_steps)

        info = {
            "sla": self.global_sla, 
            "score": float(self.score()),         
            "success": bool(self.score() >= 0.5)
        }

        # OpenEnv expects 4 elements: obs, reward, done, info
        return self._get_obs(), float(reward), done, info