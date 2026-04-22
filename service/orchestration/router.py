def classify_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()

    if "traceback" in prompt_lower or "error" in prompt_lower:
        return "explain_error"
    elif "docker" in prompt_lower:
        return "docker_debug"
    elif "sql" in prompt_lower:
        return "sql_debug"
    else:
        return "general_debug"
