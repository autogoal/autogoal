.PHONY: clean lint test-fast test-full shell docker-build docker-push

BASE_VERSION := 3.6
ALL_VERSIONS := 3.6

notebook:
	PYTHON_VERSION=${BASE_VERSION} docker-compose up

test-fast:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester make dev-test-fast

shell:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester bash

lock:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry lock

build:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry build

clean:
	git clean -fxd

lint:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester poetry run pylint autogoal

test-full:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose run make dev-test-full;)

docker-build:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose build;)

docker-push:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose push;)

# Below are the commands that will be run INSIDE the development environment, i.e., inside Docker or Travis
# These commands are NOT supposed to be run by the developer directly, and will fail to do so.

.PHONY: dev-ensure dev-build dev-install dev-test-fast dev-test-full dev-cov

dev-ensure:
	# Check if you are inside a development environment
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

dev-install: dev-ensure
	pip install poetry
	poetry config virtualenvs.create false
	poetry install
	pip install tensorflow==1.14

dev-test-fast: dev-ensure
	python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest --doctest-modules --cov=autogoal --cov-report=term-missing -v

dev-test-full: dev-ensure
	python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest --doctest-modules --cov=autogoal --cov-report=xml

dev-cov: dev-ensure
	python -m codecov
