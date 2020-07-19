![AutoGOAL Logo](docs/autogoal-banner.png)

[<img alt="PyPI" src="https://img.shields.io/pypi/v/autogoal">](https://pypi.org/project/autogoal/) [<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autogoal">](https://pypi.org/project/autogoal/) [<img alt="PyPI - License" src="https://img.shields.io/pypi/l/autogoal">](https://autogoal.github.io/contributing) [<img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/auto_goal?label=Follow%20us%20in%20Twitter%21&style=social">](https://twitter.com/auto_goal)

# AutoGOAL

> Automatic Generation, Optimization And Artificial Learning

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for _Automated Machine Learning_ (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

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
automl = AutoML()
automl.fit(X, y)
```

Sensible defaults are defined for each of the many parameters of `AutoML`.
Make sure to [read the documentation](https://autogoal.github.io/guide/) for more information on the parameters.

## Installation

Installation is very simple:

    pip install autogoal

However, `autogoal` comes with a bunch of optional dependencies. You can install them all with:

    pip install autogoal[all]

To fine-pick which dependencies you want, read the [dependencies section](https://autogoal.github.io/dependencies/).

### Using Docker 

The easiest way to get AutoGOAL up and running with all the dependencies to pull the development Docker image, which is somewhat big:

    docker pull autogoal/autogoal

If you want GPU support, pull this image instead:

    docker pull autogoal/autogoal:gpu

|  |  |
|--|--|
**CPU image:** | [<img alt="Docker Image Size (CPU)" src="https://img.shields.io/docker/image-size/autogoal/autogoal/latest">](https://hub.docker.com/r/autogoal/autogoal)
**GPU image:** | [<img alt="Docker Image Size (GPU)" src="https://img.shields.io/docker/image-size/autogoal/autogoal/gpu">](https://hub.docker.com/r/autogoal/autogoal)

Instructions for setting up Docker are available [here](https://www.docker.com/get-started).

### Installing from source

To install AutoGOAL from the source code, simply navigate to the root folder and run:

    make build

This command will create the Docker image necessary to run AutoGOAL with all its dependencies.
Alternatively, you can install on a bare Python virtual environment, using [Poetry](https://python-poetry.org/):

    pip install poetry
    poetry install

## Demo

An online demo app is available at [autogoal.github.io/demo](https://autogoal.github.io/demo).
This app showcases the main features of AutoGOAL in interactive case studies.

To run the demo locally, simply type:

    docker run -p 8501:8501 autogoal/autogoal

And navigate to [localhost:8501](http://localhost:8501).

## Documentation

This documentation is available online at [autogoal.github.io](https://autogoal.github.io). Check the following sections:

- [**User Guide**](https://autogoal.github.io/guide/): Step-by-step showcase of everything you need to know to use AuoGOAL.
- [**Examples**](https://autogoal.github.io/examples/): the best way to learn how to use AutoGOAL by practice.
- [**API**](https://autogoal.github.io/api/autogoal): details about the public API for AutoGOAL.

The HTML version can be deployed offline by downloading the AutoGOAL Docker image and running:

    docker run -p 8000:8000 autogoal/autogoal mkdocs serve -a 0.0.0.0:8000

And navigating to [localhost:8000](http://localhost:8000).

## Academic Papers

If you use AutoGOAL in academic research, please cite the following paper (to appear):

```bibtex
@article{estevez2020,
    title="General-purpose Hierarchical Optimisation of Machine Learning Pipelines with Grammatical Evolution",
    author="Suilan Estevez-Velarde and Yoan Gutiérrez and Yudivián Almeida-Cruz and Andrés Montoyo",
    journal="Information Sciences"
}
```

The technologies and theoretical results leading up to AutoGOAL have been presented at different venues:

- [Optimizing Natural Language Processing Pipelines: Opinion Mining Case Study](https://link.springer.com/chapter/10.1007/978-3-030-33904-3_15) marks the inception of the idea of using evolutionary optimization with a probabilistic search space for pipeline optimization.

- [AutoML Strategy Based on Grammatical Evolution: A Case Study about Knowledge Discovery from Text](https://www.aclweb.org/anthology/P19-1428/) applied probabilistic grammatical evolution with a custom-made grammar in the context of entity recognition in medical text.

- [General-purpose Hierarchical Optimisation of Machine Learning Pipelines with Grammatical Evolution]() (to appear) presents a more uniform framework with different grammars in different problems, from tabular datasets to natural language processing.

- [Solving Heterogeneous AutoML Problems with AutoGOAL](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_20.pdf) is the first actual description of AutoGOAL as a framework, unifying the ideas presented in the previous papers.

## Contribution

Code is licensed under MIT. Read the details in the [collaboration section](https://autogoal.github.io/contributing).
