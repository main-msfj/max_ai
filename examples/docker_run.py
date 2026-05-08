from docker_tools import add

ref = add.docker_ref()

payload = {
    "tool_ref": ref.model_dump(mode="json"),
    "record": {
        "tool_name": "add",
        "parameters": {"a": 2, "b": 3},
    },
    "context": {
        "run_id": "run_1",
        "session_id": "session_1",
        "retry_count": 0,
        "deps": {},
    },
}

print(payload)
