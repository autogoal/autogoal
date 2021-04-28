# =====================
# GPU base image
# ---------------------

FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install -U pip
RUN pip install tensorflow_addons==0.9.1
RUN pip install torch torchvision

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

COPY bash.bashrc /etc
RUN chmod +x /etc/bash.bashrc

ENV BUILD_ENVIRONMENT="development"

WORKDIR /autogoal

COPY pyproject.toml poetry.lock makefile /autogoal/

# Use system's Python for installing dev tools
RUN make env
RUN poetry install -E dev -E contrib

COPY ./ /autogoal

# Download all necessary contrib data
RUN python -m autogoal contrib download nltk
RUN python -m autogoal contrib download transformers
RUN python -m autogoal data download all

RUN make test-full
