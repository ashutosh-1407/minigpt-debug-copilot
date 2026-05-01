import re


def should_use_python_tool(prompt: str) -> bool:
    prompt_lower = prompt.lower()

    trigger_phrases = [
        "what does this return",
        "what is the output",
        "evaluate this",
        "run this",
        "compute"
    ]
    
    if any(phrase in prompt_lower for phrase in trigger_phrases):
        return True

    if "```" in prompt:
        return True

    if any(op in prompt for op in ["+", "-", "*", "/", "len(", "sum(", "max(", "min("]):
        return True

    return False

def should_return_tool_output_directly(prompt: str) -> bool:
    prompt_lower = prompt.lower()

    direct_phrases = [
        "what does this return",
        "what is the output",
        "evaluate this",
        "compute",
    ]

    return any(phrase in prompt_lower for phrase in direct_phrases)

def extract_python_code(prompt: str) -> str:
    # Case 1: fenced Python block
    fenced_match = re.search(r"```python\s*(.*?)```", prompt, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Case 2: text like "What does this return: len([1,2,3])?"
    if ":" in prompt:
        candidate = prompt.split(":", 1)[1].strip()
        return candidate.rstrip("?").strip()

    # Case 3: fallback
    return prompt.strip().rstrip("?")
