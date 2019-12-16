FROM python:3.6

ADD . /autogoal

# ENV PIPENV_VENV_IN_PROJECT="1"
# ENV XDG_CACHE_HOME="/autogoal/.venv/.cache"
# ENV POETRY_VIRTUALENVS_PATH=/autogoal/.venv

WORKDIR /autogoal

RUN make install
