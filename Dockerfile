ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}
ARG HLINK_EXTRAS=dev

RUN apt-get update && apt-get install default-jre-headless -y

RUN mkdir /hlink
WORKDIR /hlink

COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -e .[${HLINK_EXTRAS}]
