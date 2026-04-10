import os
import sys

# Support importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.http_server import create_app
from env import IncidentCommanderEnv
from models import SREAction, SREObservation

# Create the FastMCP/OpenAPI app directly via OpenEnv handler
app = create_app(
    entry_point=IncidentCommanderEnv,
    action_type=SREAction,
    observation_type=SREObservation,
    env_name="commander_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
