# First steps

AutoGOAL is a framework for the automatic generation and optimization of software pipelines.
A pipeline is defined as a series of steps, which together form a program that performs some desired task.
AutoGOAL was designed specifically for optimizing machine learnign pipelines, a problem often called [AutoML](https://automl.org),
but it can be used to optimize anything that can be defined as a set of steps with parameters.

Let's begin with the simplest possible example. Suppose we have a single function in Python that takes some input and produces some output.