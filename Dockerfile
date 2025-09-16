FROM python:3.11-slim-bullseye AS develop

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y \
        jq curl wget git netcat-traditional &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

FROM develop as runtime

WORKDIR /app

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]