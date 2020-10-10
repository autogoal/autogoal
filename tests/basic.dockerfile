FROM python:3.6

ENV BUILD_ENVIRONMENT="development"

WORKDIR /home/coder/autogoal

COPY pyproject.toml poetry.lock makefile /home/coder/autogoal/

RUN make env
RUN poetry install --extras dev

COPY . /home/coder/autogoal/
RUN make test-core
