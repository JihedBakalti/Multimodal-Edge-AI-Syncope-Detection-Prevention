def handle_agent_response(response):
    if response.get("type") == "action":
        return {"type":"action","name":response["name"],"args":response.get("args",{})}
    return {"type":"speech","text":response.get("text","")}
