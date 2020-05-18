ARG BACKEND

# =====================
# CPU base image
# ---------------------

FROM python:3.6 as base-cpu

RUN pip install -U pip
RUN pip install tensorflow
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorflow_addons

# =====================
# GPU base image
# ---------------------

FROM tensorflow/tensorflow:latest-gpu-py3 as base-gpu

RUN pip install -U pip
RUN pip install tensorflow_addons
RUN pip install torch torchvision

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

FROM base-${BACKEND}

ENV BUILD_ENVIRONMENT="development"
ENV XDG_CACHE_HOME="/opt/dev/cache"

RUN apt update && apt install -y graphviz

WORKDIR /code
COPY pyproject.toml poetry.lock makefile /code/

# Use system's Python for installing dev tools
RUN make dev-install

EXPOSE 8501
EXPOSE 8000

COPY ./ /code

RUN ln -s /code/autogoal /usr/lib/python3/dist-packages/

CMD [ "python", "-m", "autogoal", "demo" ]
