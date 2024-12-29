import logging

from abc import ABC, abstractmethod

class LLMClient(ABC):
    """LLMClient
    
    Abstract class that is used as a parent class for the higher level LLM client classes.
    
    Defines abstract classes for the following functionalities:
    - LLM managing (loading and configuration).
    - Response handling.
    - Error handling.
    """
    @abstractmethod
    def __init__(self, client_name: str, logging_level: int = logging.INFO):
        """LLMClient constructor method
        
        Configures the logging settings for the object.
        
        Args:
            client_name (str): How the module will be identified by the logger.
            logging_level (int): Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self._config = {}
        self._model_name = ""
        
        logger_formatter = logging.Formatter('[%(asctime)s : %(levelname)s : %(name)s] %(message)s')
        stderr_handle = logging.StreamHandler()
        stderr_handle.setFormatter(logger_formatter)
        self.logger = logging.getLogger(name=client_name)
        self.logger.addHandler(stderr_handle)
        self.logger.setLevel(logging_level)
        self.logger.info(f"Instantiated LLMClient '{client_name}'")
    
    def load_llm(self, model_name: str, **kwargs):
        """LLM loader method
        
        Loads the LLM model.
        
        Args:
            model_name (str): Loaded model name.
            kwargs (Dict[str, Any]): Additional arguments.
        """
        self._model_name = model_name
    
    def config(self, **kwargs):
        """LLM configuration method
        
        Configures the LLM API.
        
        Args:
            kwargs (Dict[str, Any]): Dictionary used to update the configuration.
        """
        self._config.update(kwargs)
    
    @abstractmethod
    def get_response(self, prompt: str, **kwargs) -> str:
        """Response handler abstract method
        
        Args:
            prompt (str): User input prompt.
            kwargs (Dict[str, Any]): Additional arguments.
            
        Returns:
            str: LLM response.
        """
        pass
    
    @abstractmethod
    def error_handle(self, error: Exception) -> str:
        """Error handler abstract method
        
        Args:
            prompt (str): Raw error message.
            
        Returns:
            str: Handled error.
        """
        pass
    
def unit_test():
    try:
        client = LLMClient("TestClient")
        client.load_llm("MyLLM-mini")
        client.config({"param": "value"})
    except TypeError as e:
        print("Abstract class can't be instantiated.")

if __name__ == "__main__":
    unit_test()