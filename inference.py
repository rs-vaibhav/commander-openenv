import os
import re
from openai import OpenAI
from env import IncidentCommanderEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct") 
HF_TOKEN = os.getenv("HF_TOKEN")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    err_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={float(reward):.2f} done={str(done).lower()} error={err_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list, score: float) -> None:
    r_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={r_str} score={score:.4f}", flush=True)

class SREAgent:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        self.model_name = MODEL_NAME

    def predict(self, obs):
        api_cpu, api_lat, auth_cpu, auth_err, db_cpu, db_lat, sla = obs
        
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
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            action_str = res.choices[0].message.content.strip()
            numbers = re.findall(r'\b[01234]\b', action_str)
            return int(numbers[-1]) if numbers else 0
        except Exception:
            return 0

def run_task(env, agent, task_name, max_steps):
    log_start(task=task_name, env="SRE-IncidentCommander", model=MODEL_NAME)
    env.task_name = task_name
    obs, _ = env.reset()
    rewards = []
    
    action_map = {0: "observe()", 1: "restart_api()", 2: "rollback_auth()", 3: "scale_db()", 4: "clear_cache()"}
    
    for step in range(1, max_steps + 1):
        action = agent.predict(obs)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        log_step(step=step, action=action_map.get(action, "observe()"), reward=reward, done=done)
        if done: break

    final_score = info.get('score', 0.01)
    success = info.get('success', False)
    log_end(success=success, steps=step, rewards=rewards, score=final_score)

if __name__ == "__main__":
    env = IncidentCommanderEnv()
    agent = SREAgent()
    run_task(env, agent, "easy-api-compliance", 5)
    run_task(env, agent, "medium-auth-spike", 10)
    run_task(env, agent, "hard-cascading-failure", 15)