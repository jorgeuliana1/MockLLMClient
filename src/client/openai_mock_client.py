from typing import Any, Dict, List, Optional
import time
import json
from pydantic import BaseModel
from .llm_client import LLMClient

class OpenAIError(Exception):
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

class ChoiceMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: ChoiceMessage
    finish_reason: str
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    id: str
    temperature: float
    object: str
    created: int
    model: str
    usage: Usage
    choices: List[Choice]

class OpenAIMockClient(LLMClient):
    """OpenAIMockClient
    
    Mock class to simulate OpenAI's response and error handling.
    """
    def __init__(self):
        """OpenAIMockClient constructor method
        
        Configures the logging settings for the object.
        """
        super().__init__("OpenAIMockClient")

    def get_response(self, prompt: str, temperature: float, force_error: bool = False, **kwargs) -> OpenAIResponse:
        """Response handler method
        
        Args:
            prompt (str): User input prompt.
            temperature (float): LLM temperature.
            force_error (bool): Whether to force an exception raise or not (for simulation purposes).
            kwargs (Dict[str, Any]): Additional arguments.
            
        Returns:
            OpenAIResponse: LLM response.
        """
        if force_error:
            raise OpenAIError("Forced OpenAI Error", 404, "invalid_request_error", "forced_error", None)
        
        response = {
            "id": "chat_id",
            "temperature": temperature,
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
        return OpenAIResponse(**response)
        
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
    client.config(arg="value")

    response = client.get_response("My test prompt.", temperature=0.7)
    print(response.json(indent=2))
    
    try:
        client.get_response("", force_error=True)
    except OpenAIError as e:
        error = client.error_handle(e)
        print(json.dumps(error, indent=2))

if __name__ == "__main__":
    unit_test()