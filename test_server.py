from env import IncidentCommanderEnv

try:
    e = IncidentCommanderEnv()
    obs, info = e.reset()
    print("Reset successful!")
    print(obs)
    print(info)
except Exception as ex:
    import traceback
    traceback.print_exc()
