import json
from typing import Any, Dict, Optional
from pydantic import BaseModel
from .llm_client import LLMClient

class AnthropicError(Exception):
    def __init__(self, message: str, num: int, error_type: str, code: str, param: Optional[Any] = None):
        error_msg_template = (
            "Error code {num} - {{\"error\": {{\"message\": \"{message}\", \"type\": \"{error_type}\", \"param\": \"{param}\", \"code\": \"{code}\"}}}}"
        )
        error_msg = error_msg_template.format(
            num=num,
            message=message,
            error_type=error_type,
            param=repr(param),
            code=code
        )
        super().__init__(error_msg)

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class AnthropicResponse(BaseModel):
    completion: str
    stop_reason: str
    temperature: float
    model: str
    usage: Usage

class AnthropicMockClient(LLMClient):
    def __init__(self):
        """AnthropicMockClient constructor method
        
        Configures the logging settings for the object.
        """
        super().__init__("AnthropicMockClient")

    def get_response(self, prompt: str, temperature: float, force_error: bool = False, **kwargs) -> AnthropicResponse:
        """Response handler method
        
        Args:
            prompt (str): User input prompt.
            temperature (float): LLM temperature.
            force_error (bool): Whether to force an exception raise or not (for simulation purposes).
            kwargs (Dict[str, Any]): Additional arguments.
            
        Returns:
            AnthropicResponse: LLM response.
        """
        if force_error:
            raise AnthropicError("Forced Anthropic Error", 404, "invalid_request_error", "forced_error", None)
        
        response = {
            "completion": f"This is the response from Anthropic API for the prompt: '{prompt}'.",
            "stop_reason": "stop",
            "temperature": temperature,
            "model": self._model_name,
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 32,
                "total_tokens": 48
            }
        }
        
        self.logger.info(f"API response: '{response}'")
        return AnthropicResponse(**response)

    def error_handle(self, error: AnthropicError) -> Dict[str, Any]:
        """Error handler method
        
        Args:
            error (OpenAIError): Exception object to be handled.
            
        Returns:
            Dict[str, Any]: Error dictionary.
        """
        try:
            error_json_str = str(error).split("- ", 1)[1]
            error_json = json.loads(error_json_str)
        except (IndexError, json.JSONDecodeError) as e:
            error_json = {"error": f"An unknown error occurred: '{e}'"}
        
        self.logger.error(f"An error has occurred: {error_json}")
        return error_json

def unit_test():
    client = AnthropicMockClient()
    client.load_llm("claude-3-5-sonnet-20241022")
    client.config(arg="value")

    response = client.get_response("My test prompt.", temperature=0.7)
    print(response.json(indent=2))

    try:
        client.get_response("", force_error=True)
    except AnthropicError as e:
        error = client.error_handle(e)
        print(json.dumps(error, indent=2))

if __name__ == "__main__":
    unit_test()