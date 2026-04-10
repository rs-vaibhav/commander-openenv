import os
import re
import requests
from openai import OpenAI

# Required Environment Variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    err_val = error if error else "null"
    done_str = "true" if done else "false"
    reward_formatted = f"{float(reward):.2f}"
    print(f"[STEP] step={step} action={action} reward={reward_formatted} done={done_str} error={err_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    r_str = ",".join(f"{float(r):.2f}" for r in rewards)
    succ_str = "true" if success else "false"
    print(f"[END] success={succ_str} steps={steps} rewards={r_str}", flush=True)

class SREAgent:
    def __init__(self):
        self.model_name = MODEL_NAME

    def predict(self, obs: dict):
        api_cpu = obs.get("api_cpu", 0)
        api_lat = obs.get("api_lat", 0)
        auth_cpu = obs.get("auth_cpu", 0)
        auth_err = obs.get("auth_err", 0)
        db_cpu = obs.get("db_cpu", 0)
        db_lat = obs.get("db_lat", 0)
        sla = obs.get("global_sla", 100)
        
        system_status = f"""
[SERVER TELEMETRY]
API Gateway: CPU {api_cpu:.1f}%, Latency {api_lat:.0f}ms
Auth Service: CPU {auth_cpu:.1f}%, Error Rate {auth_err:.1f}%
Database: CPU {db_cpu:.1f}%, Latency {db_lat:.0f}ms
Global SLA: {sla:.1f}%
"""
        prompt = f"""You are a Level 1 Site Reliability Engineer (SRE) managing a microservice architecture.
{system_status}
Available Actions:
0: Observe (Wait for more data)
1: Restart API Gateway (Fixes frozen API)
2: Rollback Auth Service (Fixes Auth error spikes)
3: Scale Up Database (Fixes DB CPU/Latency spikes)
4: Clear Redis Cache

Analyze the telemetry. If everything is healthy (Latency < 200ms, CPU < 80%, Errors < 5%), output 0.
If a service is failing, output the number of the corresponding fix.
Output exactly ONE digit. Nothing else."""

        try:
            res = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            action_str = res.choices[0].message.content.strip()
            numbers = re.findall(r'\b[01234]\b', action_str)
            return int(numbers[-1]) if numbers else 0
        except Exception as e:
            return 0

def run_task(agent, task_name: str, max_steps: int):
    log_start(task=task_name, env="SRE-IncidentCommander", model=MODEL_NAME)
    
    # Init remote env
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"options": {"task_name": task_name}})
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observation", {})
    except Exception as e:
        # If env is unreachable or fails to reset, abort with failure logs
        log_end(success=False, steps=0, rewards=[])
        return

    rewards = []
    action_map = {0: "observe", 1: "restart_api", 2: "rollback_auth", 3: "scale_db", 4: "clear_cache"}
    step = 0
    success = False
    
    for _ in range(max_steps):
        step += 1
        action_val = agent.predict(obs)
        action_str = action_map.get(action_val, "observe")
        
        try:
            resp = requests.post(f"{ENV_URL}/step", json={"action": {"action": action_val}})
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observation", {})
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            info = data.get("info", {})
            
            error_msg = data.get("error", None)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            rewards.append(reward)
            
            if done:
                success = info.get("success", False)
                break
        except Exception as e:
            log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
            break

    # info['score'] determines final success in openenv for tasks
    log_end(success=success, steps=step, rewards=rewards)

if __name__ == "__main__":
    agent = SREAgent()
    run_task(agent, "easy-api-compliance", 5)
    run_task(agent, "medium-auth-spike", 10)
    run_task(agent, "hard-cascading-failure", 15)