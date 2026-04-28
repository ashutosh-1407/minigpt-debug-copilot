from enum import Enum


class PromptRoute(str, Enum):
    PYTHON_ERROR = "python_error"
    FASTAPI_ERROR = "fastapi_error"
    DOCKER_ERROR = "docker_error"
    SQL_ERROR = "sql_error"
    TEST_ERROR = "test_error"
    GENERAL_DEBUG = "general_debug"

def classify_prompt(prompt: str) -> PromptRoute:
    prompt_lower = prompt.lower()

    if any(term in prompt_lower for term in ["traceback", "modulenotfounderror", "typeerror", "valueerror", "keyerror"]):
        return PromptRoute.PYTHON_ERROR
    
    if any(term in prompt_lower for term in ["fastapi", "uvicorn", "pydantic", "422"]):
        return PromptRoute.FASTAPI_ERROR
    
    if any(term in prompt_lower for term in ["docker", "container", "compose"]):
        return PromptRoute.DOCKER_ERROR
    
    if any(term in prompt_lower for term in ["sql", "postgres", "sqlite", "table", "column"]):
        return PromptRoute.SQL_ERROR
    
    if any(term in prompt_lower for term in ["pytest", "mock", "assertionerror"]):
        return PromptRoute.TEST_ERROR
    
    return PromptRoute.GENERAL_DEBUG
