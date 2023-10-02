ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}

RUN apt-get update && apt-get install default-jdk -y

RUN mkdir /hlink
WORKDIR /hlink

COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -e .[dev]
