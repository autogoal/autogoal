# .
# Usage: make [command]

# ---------------------------------------------------------------------------
# The following commands must be run OUTSIDE the development environment.
# ---------------------------------------------------------------------------

# help         Show this information.
.PHONY: help
help:
	cat makefile | grep -oP "^# \K(.*)"

# docker       Builds the development image from scratch.
.PHONY: docker
docker:
	docker build -t autogoal/autogoal:latest .

# hub    Push the development image to Docker Hub.
.PHONY: hub
hub:
	docker push autogoal/autogoal:latest

# shell        Opens a shell in the development image.
.PHONY: shell
shell:
	docker-compose run autogoal bash

# docs         Compile and publish the documentation to Github.
.PHONY: docs
docs:
	docker run --rm -it -u $(id -u):$(id -g) -v `pwd`:/code -v `pwd`/autogoal:/usr/local/lib/python3.6/site-packages/autogoal --network host autogoal/autogoal:latest bash -c "python /code/docs/make_docs.py && mkdocs build"
	(cd site && rm -rf .git && git init && git remote add origin git@github.com:autogoal/autogoal.github.io && git add . && git commit -a -m "Update docs" && git push -f origin master)

# ---------------------------------------------------------------------------
# The following commands must be run INSIDE the development environment.
# ---------------------------------------------------------------------------

.PHONY: ensure-dev
ensure-dev:
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

# env          Setup the development environment.
.PHONY: env
env: ensure-dev
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
	ln -s ${HOME}/.poetry/bin/poetry /usr/bin/poetry
	poetry config virtualenvs.create false

# install      Install all the development dependencies.
.PHONY: install
install: ensure-dev
	poetry install

# test         Run the minimal unit tests (not marked slow).
.PHONY: test
test: ensure-dev
	python -m pytest autogoal tests --doctest-modules -m "not slow" --ignore=autogoal/contrib/torch --ignore=autogoal/_old --cov=autogoal --cov-report=term-missing -v

# test-full    Run all unit tests including the slow ones.
.PHONY: test-full
test-full: ensure-dev
	python -m pytest autogoal tests --doctest-modules --ignore=autogoal/contrib/torch --ignore=autogoal/_old --cov=autogoal --cov-report=term-missing -v

# cov          Run the coverage analysis.
.PHONY: cov
cov: ensure-dev
	python -m codecov
