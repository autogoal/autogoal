# AutoGOAL

> Automatic Generation, Optimization And Learning

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for Automatic Machine Learning (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

## Notes for ICML reviewers

The following documentation has been modified and anonymized to serve as complementary resources
for the review of the paper `AutoGOAL: Automatic Discovery of Heterogeneous Machine Learning Pipelines`.

These materials show additional information about the use of AutoGOAL, implementation details such
as the class hierarchy of data types and the list of available algorithms, and experimentation details.
A simplified PDF version is bundled along with the source code for AutoGOAL.
This version only contains the most relevant part of the documentation for the conference purposes.
Additionally, a video of a Streamlit-based demo for AutoGOAL is also provided to illustrate a very simplified use case.

## Installation

The easiest way to get AutoGOAL up and running right now is to pull the development Docker image:

    docker pull autogoal/autogoal

Instructions for setting up Docker are available [here](https://www.docker.com/get-started).

!!! note
    A proper Python package is in the making. However, for anonymity reasons
    this package will not published until the anonymity period for ICML 2020 ends.

### Installing from source

To install AutoGOAL from the source code, simply navigate to the root folder and run:

    make docker-build

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

The core of this functionality lies in the [`AutoML`](./api/autogoal.ml#automl) class.

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

!!! note
    This documentation will be publicly hosted online once the anonymity period for ICML 2020 ends.

### User Guide

The step-by-step [User Guide](./guide/) will show you everything you need to know to use AuoGOAL.

### Examples

Looking at the [examples](./examples/) is the best way to learn how to use AutoGOAL.

### API

The [API documentation](./api) details the public API for AutoGOAL.

## Contribution

Code is licensed under MIT. Read the details in the [collaboration section](./contributing).
