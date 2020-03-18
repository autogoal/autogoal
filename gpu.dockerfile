# =====================
# Generic build system
# ---------------------

ARG PYTHON_VERSION

FROM tensorflow/tensorflow:latest-gpu-py3

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

WORKDIR /code
COPY pyproject.toml poetry.lock makefile /code/

ENV BUILD_ENVIRONMENT="development"
ENV XDG_CACHE_HOME="/opt/dev/cache"

# Use system's Python for installing dev tools
RUN make dev-install

# RUN pip install tensorflow==1.14
RUN pip3 install torch torchvision

RUN apt update && apt install -y graphviz

EXPOSE 8501
EXPOSE 8000

COPY ./ /code

CMD [ "python", "-m", "autogoal", "demo" ]
