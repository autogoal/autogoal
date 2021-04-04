![AutoGOAL Logo](https://autogoal.github.io/autogoal-banner.png)

[<img alt="PyPI" src="https://img.shields.io/pypi/v/autogoal">](https://pypi.org/project/autogoal/) [<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autogoal">](https://pypi.org/project/autogoal/) [<img alt="PyPI - License" src="https://img.shields.io/pypi/l/autogoal">](https://autogoal.github.io/contributing) [<img alt="GitHub stars" src="https://img.shields.io/github/stars/autogoal/autogoal?style=social">](https://github.com/autogoal/autogoal/stargazers) [<img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/auto_goal?label=Followers&style=social">](https://twitter.com/auto_goal)

[<img alt="GitHub Workflow Status (branch)" src="https://img.shields.io/github/workflow/status/autogoal/autogoal/CI/main?label=unit tests&logo=github">](https://github.com/autogoal/autogoal/actions)
[<img src="https://codecov.io/gh/autogoal/autogoal/branch/main/graph/badge.svg" />](https://codecov.io/gh/autogoal/autogoal/)
[<img alt="Docker Image Size (CPU)" src="https://img.shields.io/docker/image-size/autogoal/autogoal/latest">](https://hub.docker.com/r/autogoal/autogoal)
[<img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/autogoal/autogoal">](https://hub.docker.com/r/autogoal/autogoal)

# AutoGOAL

> Automatic Generation, Optimization And Artificial Learning

AutoGOAL is a Python library for automatically finding the best way to solve a given task.
It has been designed mainly for _Automated Machine Learning_ (aka [AutoML](https://www.automl.org))
but it can be used in any scenario where you have several possible ways to solve a given task.

Technically speaking, AutoGOAL is a framework for program synthesis, i.e., finding the best program to solve
a given problem, provided that the user can describe the space of all possible programs.
AutoGOAL provides a set of low-level components to define different spaces and efficiently search in them.
In the specific context of machine learning, AutoGOAL also provides high-level components that can be used as a black-box in almost any type of problem and dataset format.

## ⭐ Quickstart

AutoGOAL is first and foremost a framework for Automated Machine Learning.
As such, it comes pre-packaged with hundreds of low-level machine learning
algorithms that can be automatically assembled into pipelines for different problems.

The core of this functionality lies in the [`AutoML`](https://autogoal.github.io/api/autogoal.ml#automl) class.

To illustrate the simplicity of its use we will load a dataset and run an automatic classifier in it.
The following code will run for approximately 5 minutes on a classic dataset.

```python
from autogoal.datasets import cars
from autogoal.kb import (MatrixContinuousDense, 
                         Supervised, 
                         VectorCategorical)
from autogoal.ml import AutoML

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousDense, 
           Supervised[VectorCategorical]),
    output=VectorCategorical
)

# Run the pipeline search process
automl.fit(X, y)

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)
```

Sensible defaults are defined for each of the many parameters of `AutoML`.
Make sure to [read the documentation](https://autogoal.github.io/guide/) for more information.

## ⚙️ Installation

The easiest way to get AutoGOAL up and running with all the dependencies is to pull the development Docker image, which is somewhat big:

    docker pull autogoal/autogoal

Instructions for setting up Docker are available [here](https://www.docker.com/get-started).

Once you have the development image downloaded, you can fire up a console and use AutoGOAL interactively.

![](https://autogoal.github.io/shell.svg)

If you prefer to not use Docker, or you don't want all the dependencies, you can also install AutoGOAL directly with pip:

    pip install autogoal

This will install the core library but you won't be able to use any of the underlying machine learning algorithms until you install the corresponding optional dependencies. You can install them all with:

    pip install autogoal[contrib]

To fine-pick which dependencies you want, read the [dependencies section](https://autogoal.github.io/dependencies/).

> ⚠️ **NOTE**: By installing through `pip` you will get the latest release version of AutoGOAL, while by installing through Docker, you will get the latest development version. 
>
> The development version is mostly up-to-date with the `main` branch, hence it will probably contain more features, but also more bugs, than the release version.

## 💻 CLI

You can use AutoGOAL directly from the CLI. To see options just type:

    autogoal

Using the CLI you can train and use AutoML models, download datasets and inspect the contrib libraries without writing a single line of code.

![](https://autogoal.github.io/shell/autogoal_cli.svg)

Read more in the [CLI documentation](https://autogoal.github.io/cli).

## 🤩 Demo

An online demo app is available at [autogoal.github.io/demo](https://autogoal.github.io/demo).
This app showcases the main features of AutoGOAL in interactive case studies.

To run the demo locally, simply type:

    docker run -p 8501:8501 autogoal/autogoal

And navigate to [localhost:8501](http://localhost:8501).

## ⚖️ API stability

We make a conscious effort to maintain a consistent public API across versions, but the private API can change at any time.
In general, everything you can import from `autogoal` without underscores is considered public. 

For example:

```python
# "clean" imports are part of the public API
from autogoal import optimize   
from autogoal.ml import AutoML  
from autogoal.contrib.sklearn import find_classes

# public members of public types as well
automl = AutoML
automl.fit(...) 

# underscored imports are part of the private API
from autogoal.ml._automl import ...
from autogoal.contrib.sklearn._generated import ...

# as well as private members of any type
automl._input_type(...)

```

These are our consistency rules:

- Major breaking changes are introduced between major version updates, e.g., `x.0` and `y.0`. These can be additions, removals, or modifications of any kind in any part of the API.

- Between minor version updates, e.g., `1.x` and `1.y`, you can expect to find new functionality, but anything you can use from the   public API will still be there with a consistent semantic (save for bugfixes).

- Between micro version updates, e.g., `1.3.x` and `1.3.y`, the public API is frozen even for additions.

- The private API can be changed at all times.

⚠️ While AutoGOAL is on public beta (versions `0.x`) the public API is considered unstable and thus everything can change. However, we try to keep breaking changes to a minimum.

## 📚 Documentation

This documentation is available online at [autogoal.github.io](https://autogoal.github.io). Check the following sections:

- [**User Guide**](https://autogoal.github.io/guide/): Step-by-step showcase of everything you need to know to use AuoGOAL.
- [**Examples**](https://autogoal.github.io/examples/): The best way to learn how to use AutoGOAL by practice.
- [**API**](https://autogoal.github.io/api/autogoal): Details about the public API for AutoGOAL.

The HTML version can be deployed offline by downloading the [AutoGOAL Docker image](https://hub.docker.com/autogoal/autogoal) and running:

    docker run -p 8000:8000 autogoal/autogoal mkdocs serve -a 0.0.0.0:8000

And navigating to [localhost:8000](http://localhost:8000).

## 📃 Publications

If you use AutoGOAL in academic research, please cite the following paper:

```bibtex
@article{estevez2020general,
  title={General-purpose hierarchical optimisation of machine learning pipelines with grammatical evolution},
  author={Est{\'e}vez-Velarde, Suilan and Guti{\'e}rrez, Yoan and Almeida-Cruz, Yudivi{\'a}n and Montoyo, Andr{\'e}s},
  journal={Information Sciences},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.ins.2020.07.035}
}
```

The technologies and theoretical results leading up to AutoGOAL have been presented at different venues:

- [Optimizing Natural Language Processing Pipelines: Opinion Mining Case Study](https://link.springer.com/chapter/10.1007/978-3-030-33904-3_15) marks the inception of the idea of using evolutionary optimization with a probabilistic search space for pipeline optimization.

- [AutoML Strategy Based on Grammatical Evolution: A Case Study about Knowledge Discovery from Text](https://www.aclweb.org/anthology/P19-1428/) applied probabilistic grammatical evolution with a custom-made grammar in the context of entity recognition in medical text.

- [General-purpose Hierarchical Optimisation of Machine Learning Pipelines with Grammatical Evolution](https://doi.org/10.1016/j.ins.2020.07.035) presents a more uniform framework with different grammars in different problems, from tabular datasets to natural language processing.

- [Solving Heterogeneous AutoML Problems with AutoGOAL](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_20.pdf) is the first actual description of AutoGOAL as a framework, unifying the ideas presented in the previous papers.

## 🤝 Contribution

Code is licensed under MIT. Read the details in the [collaboration section](https://autogoal.github.io/contributing).

This project follows the [all-contributors](https://allcontributors.org) specification. Any contribution will be given credit, from fixing typos, to reporting bugs, to implementing new core functionalities. 

Here are all the current contributions. 

> **🙏 Thanks!**

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/sestevez"><img src="https://avatars3.githubusercontent.com/u/6156391?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Suilan Estevez-Velarde</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=sestevez" title="Code">💻</a> <a href="https://github.com/autogoal/autogoal/commits?author=sestevez" title="Tests">⚠️</a> <a href="#ideas-sestevez" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/autogoal/autogoal/commits?author=sestevez" title="Documentation">📖</a></td>
    <td align="center"><a href="https://apiad.net"><img src="https://avatars3.githubusercontent.com/u/1778204?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alejandro Piad</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=apiad" title="Code">💻</a> <a href="https://github.com/autogoal/autogoal/commits?author=apiad" title="Tests">⚠️</a> <a href="https://github.com/autogoal/autogoal/commits?author=apiad" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/yudivian"><img src="https://avatars1.githubusercontent.com/u/5324359?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yudivián Almeida Cruz</b></sub></a><br /><a href="#ideas-yudivian" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/autogoal/autogoal/commits?author=yudivian" title="Documentation">📖</a></td>
    <td align="center"><a href="http://orcid.org/0000-0002-4052-7427"><img src="https://avatars2.githubusercontent.com/u/25705914?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ygutierrez</b></sub></a><br /><a href="#ideas-joogvzz" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/autogoal/autogoal/commits?author=joogvzz" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/EEstevanell"><img src="https://avatars0.githubusercontent.com/u/45082075?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ernesto Luis Estevanell Valladares</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=EEstevanell" title="Code">💻</a> <a href="https://github.com/autogoal/autogoal/commits?author=EEstevanell" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://alexfertel.netlify.app"><img src="https://avatars3.githubusercontent.com/u/22298999?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alexander Gonzalez</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=alexfertel" title="Code">💻</a> <a href="https://github.com/autogoal/autogoal/commits?author=alexfertel" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/anshu-trivedi-501a7b146/"><img src="https://avatars1.githubusercontent.com/u/47869948?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Anshu Trivedi</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=AnshuTrivedi" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://alxrcs.github.io"><img src="https://avatars1.githubusercontent.com/u/8171561?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alex Coto</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/commits?author=alxrcs" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/geblanco"><img src="https://avatars3.githubusercontent.com/u/6652222?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Guillermo Blanco</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/issues?q=author%3Ageblanco" title="Bug reports">🐛</a> <a href="https://github.com/autogoal/autogoal/commits?author=geblanco" title="Code">💻</a> <a href="https://github.com/autogoal/autogoal/commits?author=geblanco" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/yacth"><img src="https://avatars3.githubusercontent.com/u/71322097?v=4?s=100" width="100px;" alt=""/><br /><sub><b>yacth</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/issues?q=author%3Ayacth" title="Bug reports">🐛</a> <a href="https://github.com/autogoal/autogoal/commits?author=yacth" title="Code">💻</a></td>
    <td align="center"><a href="https://sourceplusplus.com"><img src="https://avatars0.githubusercontent.com/u/3278877?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Brandon Fergerson</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/issues?q=author%3ABFergerson" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://adityanikhil.github.io/main/"><img src="https://avatars2.githubusercontent.com/u/30192967?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Aditya Nikhil</b></sub></a><br /><a href="https://github.com/autogoal/autogoal/issues?q=author%3AAdityaNikhil" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
 
