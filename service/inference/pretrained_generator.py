import torch
from model.loaders.pretrained_loader import LoadedPretrainedModel
from service.orchestration.prompt_templates import build_prompt
from service.orchestration.router import classify_prompt
from service.orchestration.tool_router import should_use_python_tool, should_return_tool_output_directly, extract_python_code
from model.utils import get_device
from service.mcp_server.tools.python_executor import execute_python_code


class PretrainedModelGenerator:
    def __init__(self, loaded_model: LoadedPretrainedModel):
        self.device = get_device()
        self.loaded_model = loaded_model

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.loaded_model.settings.default_max_new_tokens

        route = classify_prompt(prompt)

        tool_used = None
        tool_result = None
        if should_use_python_tool(prompt):
            tool_used = "python_executor"
            code = extract_python_code(prompt)
            tool_result = execute_python_code(code)
            if tool_result["success"]:
                tool_output = tool_result["output"]
            else:
                tool_output = f"ERROR: {tool_result['error']}"
            
            if should_return_tool_output_directly(prompt):
                return tool_output, route, tool_used, tool_result
            
            formatted_prompt = (
                f"User: {prompt.strip()}\n\n"
                f"Tool Used: {tool_used}\n\n"
                f"Code Executed:\n{code}\n\n"
                f"Tool Output: \n{tool_output}\n\n"
                f"Assistant:"
            )
        else:
            formatted_prompt = build_prompt(
                prompt=prompt, 
                route=route
            )

        context = self.loaded_model.tokenizer(formatted_prompt, return_tensors="pt")

        with torch.no_grad():
            generated = self.loaded_model.model.generate(
                **context,
                max_new_tokens=80,
                do_sample=False,   # greedy first
                eos_token_id=self.loaded_model.tokenizer.eos_token_id,
                pad_token_id=self.loaded_model.tokenizer.eos_token_id,
            )

        decoded = self.loaded_model.tokenizer.decode(generated[0], skip_special_tokens=False)

        answer = decoded.split("Assistant:", 1)[1].strip()
        answer = answer.split("<END>")[0].strip()

        return answer, route, tool_used, tool_result
