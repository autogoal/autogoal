# AutoGOAL

> Automatic Generation, Optimization And Learning

## What is this?

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for Automatic Machine Learning (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

## Installation

The easiest way to get AutoGOAL up and running right now is to pull the development Docker image:

    docker pull autogoal/autogoal

Instructions for setting up Docker are available [here](https://www.docker.com/get-started).

!!! note
    A proper Python package is in the making. However, for anonymity reasons
    this package will not published until the anonymity period for ICML 2020 ends.

## Demo

To quickly see AutoGOAL in action, simply run:

    docker run -p 8501:8501 autogoal/autogoal

And navigate to [localhost:8501](http://localhost:8501).

## Documentation

This documentation is available in HTML and PDF versions. If you are reading the PDF version, some
features like search and navigation are unfortunately not available. The HTML version can be
deployed by downloading the AutoGOAL Docker image and running:

    docker run -p 8000:8000 autogoal/autogoal mkdocs serve

And navigating to [localhost:8000](http://localhost:8000).

!!! note
    This documentation will be publicly hosted online once the anonymity period for ICML 2020 ends.

### User Guide

The step-by-step [User Guide](/guide/quickstart) will show you everything you need to know to use AuoGOAL.

### Examples

Looking at the [examples](/examples/) is the best way to learn how to use AutoGOAL.

### API

The [API documentation](/api) details the public API for AutoGOAL.

## Contribution

Code is licensed under MIT. Read the details in the [collaboration section](/contributing).
