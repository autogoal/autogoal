# User Guide

AutoGOAL is a framework for the automatic generation and optimization of software pipelines.
A pipeline is defined as a series of steps, which together form a program that performs some desired task.
AutoGOAL was designed specifically for optimizing machine learning pipelines, a problem often called [AutoML](https://automl.org),
but it can be used to optimize anything that can be defined as a set of steps with parameters.

AutoGOAL has been designed to suit a broad range of users with different skill levels, from beginners to experts.
Likewise, the API suits different needs, from practical use cases requiring fast iteration and out-of-the-box solutions
to more involved, research-oriented use cases that require customizing and tweaking many things.
Whatever your case, the following guides should help you get started.

* **[Black-Box Optimization](./blackbox/)**:
    A black-box optimizer that can be applied to any function.

* **[Predefined Pipelines](./predefined/)**:
    Pre-packaged with pipelines based on popular machine learning frameworks,
    that you can use in few lines of code to build highly optimized machine learning pipelines for a broad range of problems.

* **[Class-based Pipelines](./cfg/)**:
    The class-based API allows you to turn any class hierarchy into an optimizable space.
    You define classes and annotate the constructor's parameters
    with attributes, and AutoGOAL automatically builds a grammar that generates all possible
    instances of your hierarchy.

* **[Graph-based Pipelines](./graph/)**:
    The graph-based API allows you to explore spaces defined as graphs. You define a graph grammar as
    a set of graph rewriting rules, that take existing nodes and replace them for more complex patterns.
    AutoGOAL then transforms into an evaluatable object, e.g., a neural network.

* **[Functional Pipelines](./functional/)**:
    If none of the previous suits you, the functional API allows you to magically turn any Python code
    that solves some task into an optimizable pipeline.
    You write a regular method and introduce AutoGOAL parameters in the code flow, which will be later
    automatically optimized to produce the optimal output.

Don't forget to also look at the [examples](../examples/) for more down-to-earth specific use cases.
