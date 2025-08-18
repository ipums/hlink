ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-bookworm
ARG HLINK_EXTRAS=dev

RUN apt-get update && apt-get install openjdk-17-jre-headless -y

RUN mkdir /hlink
WORKDIR /hlink

COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -e .[${HLINK_EXTRAS}]
