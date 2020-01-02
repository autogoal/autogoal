# AutoGOAL

> Automatic Generation, Optimization And Learning

## What is this?

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for Automatic Machine Learning (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways (i.e., programs) to solve a given task.

Looking at the [examples](autogoal/examples) is the best way to learn how to use AutoGOAL:

- [Finding the best neural network for a dataset](autogoal/examples/keras_text_classifier.py)
- [Finding the best scikit-learn pipeline for a dataset](autogoal/examples/sklearn_simple_grammar.py)

In its core, AutoGOAL is an evolutionary program synthesis framework. It consists
of two main components:

1. The [grammar](autogoal/grammar) module contains tools to design context-free and graph grammars that describe the space of all possible solutions to a specific problem.
2. The [search](autogoal/search) module contains tools for automatically exploring this vast space efficiently.
