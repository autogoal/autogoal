FROM python:3.6

ADD . /autogoal

ENV XDG_CACHE_HOME="/opt/venv/cache"
ENV POETRY_VIRTUALENVS_PATH="/opt/venv"

WORKDIR /autogoal

VOLUME [ "/opt/venv" ]

RUN make install
