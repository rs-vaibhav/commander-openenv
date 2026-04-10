from pydantic import BaseModel, Field

class SREObservation(BaseModel):
    api_cpu: float = Field(..., description="API Gateway CPU usage (%)")
    api_lat: float = Field(..., description="API Gateway latency (ms)")
    auth_cpu: float = Field(..., description="Auth Service CPU usage (%)")
    auth_err: float = Field(..., description="Auth Service error rate (%)")
    db_cpu: float = Field(..., description="Database CPU usage (%)")
    db_lat: float = Field(..., description="Database latency (ms)")
    global_sla: float = Field(..., description="Global Service Level Agreement (%)")

class SREAction(BaseModel):
    action: int = Field(..., description="Action to take. 0: Observe, 1: Restart API, 2: Rollback Auth, 3: Scale DB, 4: Clear Redis Cache")

class SREState(BaseModel):
    observation: SREObservation
    current_step: int
    incident_resolved: bool
    task_name: str
    downtime_penalty_accumulated: float
