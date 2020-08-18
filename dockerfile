# =====================
# GPU base image
# ---------------------

FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install -U pip
RUN pip install tensorflow_addons==0.9.1
RUN pip install torch torchvision

# =====================
# User stuff
# ---------------------

RUN apt-get update \
 && apt-get install -y \
    curl \
    dumb-init \
    htop \
    locales \
    man \
    nano \
    git \
    procps \
    ssh \
    sudo \
    vim \
    graphviz \
  && rm -rf /var/lib/apt/lists/*

# https://wiki.debian.org/Locale#Manually
RUN sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen \
  && locale-gen
ENV LANG=en_US.UTF-8

RUN chsh -s /bin/bash
ENV SHELL=/bin/bash

RUN adduser --gecos '' --disabled-password coder && \
  echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

ENV BUILD_ENVIRONMENT="development"
ENV XDG_CACHE_HOME="/opt/dev/cache"

WORKDIR /home/coder/autogoal

COPY pyproject.toml poetry.lock makefile /home/coder/autogoal/

# Use system's Python for installing dev tools
RUN make env
RUN make install

EXPOSE 8501
EXPOSE 8000

COPY ./ /home/coder/autogoal
RUN ln -s /home/coder/autogoal/autogoal /usr/lib/python3/dist-packages/autogoal
USER coder

CMD [ "python", "-m", "autogoal", "demo" ]
