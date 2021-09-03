# syntax=docker/dockerfile:1
FROM python:3
ENV PYTHONUNBUFFERED=1
WORKDIR /stattest
COPY requirements.txt /stattest/
RUN pip install -r requirements.txt
COPY . /stattest/
