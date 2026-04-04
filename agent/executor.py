from actions import ACTIONS

def execute_action(action_name, args):
    if action_name in ACTIONS:
        return ACTIONS[action_name](**args)
    else:
        return {"status":"failed"}
