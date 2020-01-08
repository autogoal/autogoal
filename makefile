BASE_VERSION := 3.6
ALL_VERSIONS := 3.6

.PHONY: test-fast
test-fast:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester make dev-test-fast

.PHONY: notebook
notebook:
	PYTHON_VERSION=${BASE_VERSION} docker-compose up

.PHONY: docs-serve
docs-serve:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester python /code/examples/make.py && mkdocs serve

.PHONY: docs-deploy
docs-deploy:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester python /code/examples/make.py && cp docs/index.md Readme.md && mkdocs gh-deploy

.PHONY: shell
shell:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester bash

.PHONY: lock
lock:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry lock

.PHONY: build
build:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry build

.PHONY: clean
clean:
	git clean -fxd

.PHONY: lint
lint:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry run pylint autogoal

.PHONY: test-full
test-full:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose run make dev-test-full;)

.PHONY: docker-build
docker-build:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose build;)

.PHONY: docker-push
docker-push:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose push;)

# Below are the commands that will be run INSIDE the development environment, i.e., inside Docker or Travis
# These commands are NOT supposed to be run by the developer directly, and will fail to do so.

.PHONY: dev-ensure
dev-ensure:
	# Check if you are inside a development environment
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

.PHONY: dev-install
dev-install: dev-ensure
	pip install poetry
	poetry config virtualenvs.create false
	poetry install
	pip install tensorflow==1.14
#
.PHONY: dev-test-fast
dev-test-fast: dev-ensure
	python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest --doctest-modules --ignore=notebooks --ignore=examples --ignore=docs --ignore=autogoal/_old --cov=autogoal --cov-report=term-missing -v

.PHONY: dev-test-full
dev-test-full: dev-ensure
	python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest --doctest-modules --ignore=notebooks --ignore=examples --ignore=docs --ignore=autogoal/_old --cov=autogoal --cov-report=xml

.PHONY: dev-cov
dev-cov: dev-ensure
	python -m codecov
