# =====================
# Generic build system
# ---------------------

FROM python:3.6 

# ==========================================
# Project-specific installation instruction
# ------------------------------------------

WORKDIR /code
COPY pyproject.toml poetry.lock makefile /code/

ENV BUILD_ENVIRONMENT="development"
ENV XDG_CACHE_HOME="/opt/dev/cache"

# Use system's Python for installing dev tools
RUN make dev-install

RUN pip install tensorflow
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorflow_addons

RUN apt update && apt install -y graphviz

EXPOSE 8501
EXPOSE 8000

COPY ./ /code

RUN ln -s /code/autogoal /usr/lib/python3/dist-packages/

CMD [ "python", "-m", "autogoal", "demo" ]
