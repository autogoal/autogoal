# =====================
# Generic build system
# ---------------------

ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

WORKDIR /code
COPY pyproject.toml poetry.lock makefile /code/

ENV BUILD_ENVIRONMENT="development"
ENV XDG_CACHE_HOME="/opt/dev/cache"

# Use system's Python for installing dev tools
RUN make dev-install

COPY . /code

CMD [ "bash" ]
