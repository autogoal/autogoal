# ‎‎
#     ^         _         ____  ___    ^    _     
#    / \  _   _| |_ ___  / ___|/ _ \  / \  | |    
#   / _ \| | | | __/ _ \| |_ _| | | |/ _ \ | |    
#  / ___ \ |_| | || (_) | |_| | |_| / ___ \| |___ 
# /_/   \_\__,_|\__\___/ \____|\___/_/   \_\_____|
#                                                 
# Usage: make [command]
# ‎‎
# ---------------------------------------------------------------------------
# The following commands can be run anywhere.
# ---------------------------------------------------------------------------
# ‎‎

# help         Show this information.
.PHONY: help
help:
	cat makefile | grep -oP "^# \K(.*)"

# clean        Remove (!) all untracked and ignored files.
.PHONY: clean
clean:
	git clean -xdff

# ‎‎
# ---------------------------------------------------------------------------
# The following commands must be run OUTSIDE the development environment.
# ---------------------------------------------------------------------------
# ‎‎

# docker       Builds the development image from scratch.
.PHONY: docker
docker:
	docker build -t autogoal/autogoal:latest .

# pull         Pull the development image.
.PHONY: pull
pull:
	docker pull autogoal/autogoal:latest

# push         Push the development image to Docker Hub.
.PHONY: push
push:
	docker push autogoal/autogoal:latest

# shell        Opens a shell in the development image.
.PHONY: shell
shell:
	docker-compose run autogoal bash

# demo         Run the demo in the development image.
.PHONY: demo
demo:
	docker-compose run autogoal

# mkdocs       Run the docs server in the development image.
.PHONY: mkdocs
mkdocs:
	docker-compose run autogoal mkdocs serve -a 0.0.0.0:8000

# test-ci      Test only the core code in a newly built image.
.PHONY: test-ci
test-ci:
	docker build -t autogoal:basic -f tests/basic.dockerfile .

# ‎‎
# ---------------------------------------------------------------------------
# The following commands must be run INSIDE the development environment.
# ---------------------------------------------------------------------------
# ‎‎

.PHONY: ensure-dev
ensure-dev:
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

# docs         Compile and publish the documentation to Github.
.PHONY: docs
docs: ensure-dev
	python3 -m illiterate --inline autogoal docs/api
	python3 -m illiterate --inline tests/examples docs/examples
	python3 -m illiterate --inline tests/guide docs/guide
	cp Readme.md docs/index.md
	# python3 -m typer_cli autogoal/__main__.py utils docs > docs/cli-api.md
	mkdocs build
	# (cd site && rm -rf .git && git init && git remote add origin git@github.com:autogoal/autogoal.github.io && git add . && git commit -a -m "Update docs" && git push -f origin master)

# format       Format all source code inplace using `black`.
format:
	(git status | grep "nothing to commit") && sudo black autogoal/ tests/ || echo "REFUSING TO REFORMAT WITH UNCOMMITED CHANGES"

# anim         Make CLI animations
.PHONY: anim
anim:
	termtosvg -c "bash docs/shell/autogoal_cli.sh" docs/shell/autogoal_cli.svg -g 80x20 -m 100 -t window_frame_powershell

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

# test-core    Run the core unit tests (not contrib).
.PHONY: test-core
test-core: ensure-dev
	python -m pytest autogoal tests/core --doctest-modules -m "not slow" --ignore=autogoal/contrib --ignore=autogoal/datasets --ignore=autogoal/experimental --cov=autogoal --cov-report=term-missing -v

# test-full    Run all unit tests including the (very) slow ones.
.PHONY: test-full
test-full: ensure-dev
	python -m pytest autogoal tests --doctest-modules --cov=autogoal --cov-report=term-missing -v

# cov          Run the coverage analysis.
.PHONY: cov
cov: ensure-dev
	python -m codecov

# ‎‎
