.PHONY: build clean install test lint cov

# TODO: Update your project folder
PROJECT=autogoal

build:
	pipenv run python setup.py sdist bdist_wheel

clean:
	git clean -fxd

install:
	pip install --upgrade pip
	pip install poetry
	poetry install
	poetry run pip install tensorflow==1.14.0

test:
	poetry run pytest --doctest-modules --cov=$(PROJECT) --cov-report=xml -v

lint:
	poetry run pylint $(PROJECT)

cov:
	poetry run codecov
