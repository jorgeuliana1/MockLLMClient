# LLM Client

This repository contains two mock LLM wrapper classes.

## How to use

Build the Docker image

```
docker-compose build
```

Run the Docker container

```
docker-compose run client <arguments>
```

The Docker arguments are the following

```
prompt (Obligatory)
--client (OpenAI or Anthropic)
--force-error (store_true, if present, an exception will be raised)
--model (string)
```

Example of command:
```
docker-compose run client My prompt --force-error
```

Obtained output:
```
[2024-12-29 22:31:57,773 : INFO : OpenAIMockClient] Instantiated LLMClient 'OpenAIMockClient'
[2024-12-29 22:31:57,774 : ERROR : OpenAIMockClient] An error has occurred: {'error': {'message': 'Forced OpenAI Error', 'type': 'invalid_request_error', 'param': 'None', 'code': 'forced_error'}}
```