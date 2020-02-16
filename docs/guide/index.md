# User Guide

AutoGOAL is a framework for the automatic generation and optimization of software pipelines.
A pipeline is defined as a series of steps, which together form a program that performs some desired task.
AutoGOAL was designed specifically for optimizing machine learning pipelines, a problem often called [AutoML](https://automl.org),
but it can be used to optimize anything that can be defined as a set of steps with parameters.

AutoGOAL has been designed to suit a broad range of users with different skill levels, from beginners to experts.
Likewise, the API suits different needs, from practical use cases requiring fast iteration and out-of-the-box solutions
to more involved, research-oriented use cases that require customizing and tweaking many things.
Whatever your case, the following guides should help you get started.

* **[Predefined Pipelines](/guide/predefined/)**:
    Pre-packaged with pipelines based on [scikit-learn](/api/sklearn/) and [keras](/api/keras/)
    that you can use in few lines of code to build highly optimized machine learning pipelines for a broad range of problems.

* **[Class-based Pipelines](/guide/cfg/)**:
    Turn any class hierarchy into an optimizable space. You define classes and annotate the constructor's parameters
    with attributes

* **[Graph-based Pipelines](/guide/graph/)**:
    Pre-packaged with pipelines based on [scikit-learn](/api/sklearn/) and [keras](/api/keras/)
    that you can use in few lines of code to build highly optimized machine learning for a broad range of problems.

* **[Functional Pipelines](/guide/functional/)**:
    If none of the previous suits you, the functional API allows you to magically turn any Python code
    that solves some task into an optimizable pipeline.

!!! note
    Don't forget to also look at the [examples](/examples/) for more down-to-earth specific use cases.
