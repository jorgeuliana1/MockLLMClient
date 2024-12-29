FROM python:3.10

WORKDIR /app

COPY . /app

ENTRYPOINT [ "python3", "src/main.py" ]