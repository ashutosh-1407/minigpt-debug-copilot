from service.orchestration.router import PromptRoute


TEMPLATES = {
    PromptRoute.PYTHON_ERROR: (
        "You are a Python debugging assistant. "
        "Explain the likely cause and give a concise next step.\n\n"
        "User: {prompt}\nAssistant:"
    ),
    PromptRoute.FASTAPI_ERROR: (
        "You are a backend debugging assistant specializing in FastAPI and API services. "
        "Explain the likely cause and what to check first.\n\n"
        "User: {prompt}\nAssistant:"
    ),
    PromptRoute.DOCKER_ERROR: (
        "You are a Docker and deployment debugging assistant. "
        "Explain the likely cause and give practical checks.\n\n"
        "User: {prompt}\nAssistant:"
    ),
    PromptRoute.SQL_ERROR: (
        "You are a SQL/database debugging assistant. "
        "Explain the likely schema, query, or connection issue and suggest a verification step.\n\n"
        "User: {prompt}\nAssistant:"
    ),
    PromptRoute.TEST_ERROR: (
        "You are a test debugging assistant. "
        "Explain why the test may be failing and what assertion or mock behavior to inspect.\n\n"
        "User: {prompt}\nAssistant:"
    ),
    PromptRoute.GENERAL_DEBUG: (
        "You are an engineering debugging assistant. "
        "Explain the likely cause and suggest the next debugging step.\n\n"
        "User: {prompt}\nAssistant:"
    ),
}

def build_prompt(prompt: str, route: PromptRoute) -> str:
    cleaned_prompt = prompt.strip()
    if cleaned_prompt.startswith("User:"):
        return cleaned_prompt
    return TEMPLATES[route].format(prompt=cleaned_prompt)
