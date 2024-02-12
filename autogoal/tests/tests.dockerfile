FROM conda/miniconda3:latest

RUN apt update \
 && apt install -y \
    curl \
    locales \
    nano \
    ssh \
    sudo \
    bash \
    git \
    make \
    gcc \
    build-essential \ 
    python3-dev

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

COPY bash.bashrc /etc
RUN chmod +x /etc/bash.bashrc

ENV BUILD_ENVIRONMENT="development"

WORKDIR /autogoal

COPY pyproject.toml poetry.lock makefile /autogoal/

RUN conda create -y --name autogoal python=3.9.16
SHELL ["conda", "run", "-n", "autogoal", "/bin/bash", "-c"]

# Use system's Python for installing dev tools
RUN make env
RUN poetry install
RUN poetry install -E contrib -E dev

COPY ./ /autogoal

# Download all necessary contrib data
RUN python -m autogoal contrib download nltk
RUN python -m autogoal contrib download transformers
RUN python -m autogoal data download all

RUN make test-full
