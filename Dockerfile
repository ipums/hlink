ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-bullseye

RUN apt-get update && apt-get install openjdk-11-jdk -y

RUN mkdir /hlink
WORKDIR /hlink

COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -e .[dev]
