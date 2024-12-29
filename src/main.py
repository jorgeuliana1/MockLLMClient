import argparse
import sys

from client import OpenAIMockClient, AnthropicMockClient, OpenAIError, AnthropicError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs='+')
    parser.add_argument("--client", default="OpenAI", choices=("OpenAI", "Anthropic"))
    parser.add_argument("--force-error", action='store_true')
    parser.add_argument("--model", default="gpt-4")
    return parser.parse_args()

def main():
    args = parse_args()
    
    class_name_string = f'{args.client}MockClient'
    force_error = args.force_error
    prompt = " ".join(args.prompt)
    llm_model = args.model
    
    client_obj = getattr(sys.modules[__name__], class_name_string)
    client = client_obj()
    client.load_llm(llm_model)
    client.config()
    try:
        response = client.get_response(prompt, force_error=force_error)
    except (OpenAIError, AnthropicError) as e:
        client.error_handle(e)
    
if __name__ == "__main__":
    main()