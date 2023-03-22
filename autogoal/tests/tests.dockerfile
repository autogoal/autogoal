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

RUN conda create -y --name autogoal python=3.7
SHELL ["conda", "run", "-n", "autogoal", "/bin/bash", "-c"]

# Use system's Python for installing dev tools
RUN make env
RUN conda install -y tensorflow-gpu==2.1.0 && pip install tensorflow-addons==0.9.1 torch==1.10.1 torchvision==0.11.2
RUN poetry install
RUN poetry install -E contrib -E dev

COPY ./ /autogoal

# Download all necessary contrib data
RUN python -m autogoal contrib download nltk
RUN python -m autogoal contrib download transformers
RUN python -m autogoal data download all

RUN make test-full
