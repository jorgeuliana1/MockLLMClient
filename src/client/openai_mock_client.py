from typing import Any, Dict
import time
import json

from .llm_client import LLMClient

class OpenAIError(Exception):
    def __init__(self, message, num, error_type, code, param=None):
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


class OpenAIMockClient(LLMClient):
    """OpenAIMockClient
    
    Mock class to simulate OpenAI's response and error handling.
    """
    def __init__(self):
        """OpenAIMockClient constructor method
        
        Configures the logging settings for the object.
        """
        super().__init__("OpenAIMockClient")

    def get_response(self, prompt: str, force_error: bool = False, **kwargs) -> Dict[str, Any]:
        """Response handler method
        
        Args:
            prompt (str): User input prompt.
            force_error (bool): Whether to force an exception raise or not (for simulation purposes).
            kwargs (Dict[str, Any]): Additional arguments.
            
        Returns:
            Dict[str, Any]: LLM response dictionary.
        """
        if force_error:
            raise OpenAIError("Forced OpenAI Error", 404, "invalid_request_error", "forced_error", None)
        
        response = {
            "id": "chat_id",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_name,
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 32,
                "total_tokens": 48
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"This is the response from the OpenAI API for the prompt: '{prompt}'."
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        
        self.logger.info(f"API response: '{response}'")
        
        return response
        
    def error_handle(self, error: OpenAIError) -> Dict[str, Any]:
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
    client = OpenAIMockClient()
    client.load_llm("gpt-4o-mini")
    client.config(arg = "value")
    client.get_response("My test prompt.")
    
    try:
        client.get_response("", force_error=True)
    except OpenAIError as e:
        client.error_handle(e)

if __name__ == "__main__":
    unit_test()