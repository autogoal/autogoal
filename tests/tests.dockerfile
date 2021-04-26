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

COPY pyproject.toml poetry.lock makefile /home/coder/autogoal/

# Use system's Python for installing dev tools
RUN make env
RUN poetry install -E dev -E contrib

COPY ./ /autogoal
RUN sudo ln -s /autogoal/autogoal /usr/lib/python3/dist-packages/autogoal

# Download all necessary contrib data
RUN python -m autogoal contrib download all
RUN python -m autogoal data download all

RUN make test-full
