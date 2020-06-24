# AutoGOAL

<img alt="PyPI" src="https://img.shields.io/pypi/v/autogoal"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autogoal"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/autogoal"> <img alt="Docker Image Size (tag)" src="https://img.shields.io/docker/image-size/autogoal/autogoal/latest">

> Automatic Generation, Optimization And Learning

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for Automatic Machine Learning (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

## Installation

Installation is very simple:

    pip install autogoal

However, `autogoal` comes with a bunch of optional dependencies. You can install them all with:

    pip install autogoal[all]

To fine-pick which dependencies you want, read the [dependencies section](https://autogoal.github.io/dependencies/).

The easiest way to get AutoGOAL up and running with all the dependencies to pull the development Docker image, which is somewhat big (~4.8 GB):

    docker pull autogoal/autogoal

Instructions for setting up Docker are available [here](https://www.docker.com/get-started).

### Installing from source

To install AutoGOAL from the source code, simply navigate to the root folder and run:

    make build

This command will create the Docker image necessary to run AutoGOAL with all its dependencies.
Alternatively, you can install on a bare Python virtual environment, using [Poetry](https://python-poetry.org/):

    pip install poetry
    poetry install

## Demo

To quickly see AutoGOAL in action, simply run:

    docker run -p 8501:8501 autogoal/autogoal

And navigate to [localhost:8501](http://localhost:8501).

## Quickstart

AutoGOAL is first and foremost a framework for Automatic Machine Learning.
As such, it comes pre-packaged with hundreds of low-level machine learning
algorithms that can be automatically assembled into pipelines for different problems.

The core of this functionality lies in the [`AutoML`](https://autogoal.github.io/api/autogoal.ml#automl) class.

To illustrate the simplicity of its use we will load a dataset and run an automatic classifier in it.

```python
from autogoal.datasets import cars
from autogoal.ml import AutoML

X, y = cars.load()
automl = AutoML(errors='ignore')
automl.fit(X, y)
```

Sensible defaults are defined for each of the many parameters of `AutoML`.
Make sure to read the documentation for more information on the parameters.

## Documentation

This documentation is available in HTML and PDF versions. If you are reading the PDF version, some
features like search and navigation are unfortunately not available.

The HTML version can be deployed by downloading the AutoGOAL Docker image and running:

    docker run -p 8000:8000 autogoal/autogoal mkdocs serve -a 0.0.0.0:8000

And navigating to [localhost:8000](http://localhost:8000).

### User Guide

The step-by-step [User Guide](https://autogoal.github.io/guide/) will show you everything you need to know to use AuoGOAL.

### Examples

Looking at the [examples](https://autogoal.github.io/examples/) is the best way to learn how to use AutoGOAL.

### API

The [API documentation](https://autogoal.github.io/api/autogoal) details the public API for AutoGOAL.

## Contribution

Code is licensed under MIT. Read the details in the [collaboration section](https://autogoal.github.io/contributing).
