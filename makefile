BASE_VERSION := 3.6
ALL_VERSIONS := 3.6
ENVIRONMENT := cpu

.PHONY: test-fast
test-fast:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} make dev-test-fast

.PHONY: notebook
notebook:
	PYTHON_VERSION=${BASE_VERSION} docker-compose up

.PHONY: docs
docs:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} python /code/docs/make_docs.py && mkdocs serve

.PHONY: docs-deploy
docs-deploy:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} python /code/docs/make_docs.py && cp docs/index.md Readme.md && mkdocs gh-deploy

.PHONY: shell
shell:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} bash

.PHONY: shell-gpu
shell-gpu:
	docker run --rm -it -u $(id -u):$(id -g) -v `pwd`:/code -v `pwd`/autogoal:/usr/lib/python3/dist-packages/autogoal  autogoal/autogoal:3.6-gpu bash

.PHONY: lock
lock:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} poetry lock

.PHONY: build
build:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} poetry build

.PHONY: clean
clean:
	git clean -fxd

.PHONY: lint
lint:
	PYTHON_VERSION=${BASE_VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} poetry run pylint autogoal

.PHONY: test-full
test-full:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose run autogoal-tester-${ENVIRONMENT} make dev-test-full;)

.PHONY: docker-build
docker-build:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose build;)
	docker tag autogoal-dev:${BASE_VERSION}-cpu autogoal/autogoal

.PHONY: docker-build-gpu
docker-build-gpu:
	docker build -t autogoal/autogoal:3.6-gpu -f gpu.dockerfile .

.PHONY: docker-push
docker-push:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose push;)

.PHONY: docker-pull
docker-pull:
	$(foreach VERSION, $(ALL_VERSIONS), PYTHON_VERSION=${VERSION} docker-compose pull;)

# Below are the commands that will be run INSIDE the development environment, i.e., inside Docker or Travis
# These commands are NOT supposed to be run by the developer directly, and will fail to do so.

.PHONY: dev-ensure
dev-ensure:
	# Check if you are inside a development environment
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

.PHONY: dev-install
dev-install: dev-ensure
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
	${HOME}/.poetry/bin/poetry config virtualenvs.create false
	${HOME}/.poetry/bin/poetry install

.PHONY: dev-test-fast
dev-test-fast: dev-ensure
	# python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest autogoal tests --doctest-modules -m "not slow" --ignore=autogoal/contrib/torch --ignore=autogoal/_old --cov=autogoal --cov-report=term-missing -v

.PHONY: dev-test-full
dev-test-full: dev-ensure
	# python -m mypy -p autogoal --ignore-missing-imports
	python -m pytest autogoal tests --doctest-modules --ignore=autogoal/contrib/torch --ignore=autogoal/_old --cov=autogoal --cov-report=term-missing -v

.PHONY: dev-cov
dev-cov: dev-ensure
	python -m codecov
