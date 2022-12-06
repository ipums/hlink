FROM python:3.10

RUN apt-get update && apt-get install openjdk-11-jdk -y

RUN mkdir /hlink
WORKDIR /hlink

COPY . .
RUN pip install -e .[dev]
