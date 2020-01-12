# AutoGOAL

> Automatic Generation, Optimization And Learning

## What is this?

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for Automatic Machine Learning (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

## Installation

Installation is very simple:

    pip install autogoal

However, `autogoal` comes with a bunch of optional dependencies. You can install them all with:

    pip install autogoal[all]

To fine-pick which dependencies you want, read the [dependencies section](/dependencies/).

## Documentation

### User Guide

The step-by-step [User Guide](/guide/quickstart) will show you everything you need to know to use AuoGOAL.

### Examples

Looking at the [examples](/examples/) is the best way to learn how to use AutoGOAL:

- [Finding the best scikit-learn pipeline for classifying movie reviews](/examples/sklearn_simple_grammar/)
- [Finding the best neural network for a dataset](/examples/keras_text_classifier/)

### API

The [API documentation](/api) details the public API for AutoGOAL.

In its core, AutoGOAL is an evolutionary program synthesis framework. It consists
of two main components:

1. The [grammar](autogoal/grammar) module contains tools to design context-free and graph grammars that describe the space of all possible solutions to a specific problem.
2. The [search](autogoal/search) module contains tools for automatically exploring this vast space efficiently.

## Contribution

Code is licensed under MIT. Read the details in the [collaboration section](/contributing).
