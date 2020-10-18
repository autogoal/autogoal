import collections
import inspect
from pathlib import Path
from typing import List

import pandas as pd
import typer

from autogoal.contrib import find_classes
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, CategoricalVector, MatrixCategorical
from autogoal.utils import Gb, Min
from autogoal.search import ConsoleLogger, ProgressLogger


app = typer.Typer(name="AutoGOAL")
contrib_app = typer.Typer(name="contrib")
automl_app = typer.Typer(name="automl")


app.add_typer(contrib_app)
app.add_typer(automl_app)


@app.callback()
def main():
    """
    Manage AutoGOAL directly from the CLI.
    """


@app.command()
def demo():
    """
    Launch streamlit demo.
    """

    try:
        from streamlit.bootstrap import run

        run(Path(__file__).parent / "contrib" / "streamlit" / "demo.py", "", "")
    except ImportError:
        print("(!) Too run the demo you need streamlit installed.")
        print("(!) Fix it by running `pip install autogoal[streamlit]`.")


@contrib_app.callback()
def contrib_main():
    """
    Inspect contrib libraries and algorithms.
    """


@contrib_app.command("list")
def contrib_list(
    verbose: bool = False,
    include: str = None,
    exclude: str = None,
    input: str = None,
    output: str = None,
):
    """
    List all currently available contrib algorithms.
    """
    from autogoal.contrib import find_classes

    classes = find_classes(include=include, exclude=exclude, input=input, output=output)
    classes_by_contrib = collections.defaultdict(list)
    max_cls_name_length = 0

    for cls in classes:
        max_cls_name_length = max(max_cls_name_length, len(cls.__name__))
        classes_by_contrib[str(cls).split(".")[2]].append(cls)

    typer.echo(
        f"⚙️  Found a total of {len(classes)} matching algorithms.", color="blue"
    )

    for contrib, clss in classes_by_contrib.items():
        typer.echo(f"🛠️  {contrib}: {len(clss)} algorithms.")

        if verbose:
            for cls in clss:
                sig = inspect.signature(cls.run)
                typer.echo(
                    f" 🔹 {cls.__name__.ljust(max_cls_name_length)} : {sig.parameters['input'].annotation} -> {sig.return_annotation}"
                )


@automl_app.callback()
def automl_callback():
    """
    Fit and predict with an AutoML model.
    """


def _load_dataset(format, input, ignore):
    if format is None:
        if input.suffix == ".csv":
            format = "csv"
        if input.suffix == ".json":
            format = "json"

    if format == "csv":
        dataset = pd.read_csv(input)
    elif format == "json":
        dataset = pd.read_json(input)
    else:
        raise ValueError("Input format not recognized. Must be either CSV or JSON.")

    if ignore:
        dataset = dataset.drop(columns=ignore)

    return dataset


@automl_app.command("fit")
def automl_fit(
    input: Path,
    output: Path = Path("automl.bin"),
    target: str = None,
    ignore: List[str] = None,
    evaluation_timeout: int = 5 * Min,
    memory_limit: int = 4 * Gb,
    search_timeout: int = 60 * 60,
    pop_size: int = 20,
    iterations: int = 100,
    format: str = None,
):
    try:
        dataset = _load_dataset(format, input, ignore)
    except ValueError as e:
        print(f"⚠️  Error: {str(e)}")
        return

    if target is None:
        target = dataset.columns[0]

    columns = set(dataset.columns)

    y = dataset[target].values
    X = dataset[list(columns - {target})].values

    automl = AutoML(
        output=CategoricalVector(),
        search_kwargs=dict(
            evaluation_timeout=evaluation_timeout,
            memory_limit=memory_limit,
            search_timeout=search_timeout,
            pop_size=pop_size,
        ),
        search_iterations=iterations,
    )

    print(f"🏋️  Training on {len(dataset)} items.")
    automl.fit(X, y, logger=[ConsoleLogger(), ProgressLogger()])

    with output.open("wb") as fp:
        automl.save(fp)

    print(f"💾 Saving model to {output.absolute()}.")
    

@automl_app.command("predict")
def automl_predict(
    input: Path,
    model: Path = Path("automl.bin"),
    output: Path = None,
    ignore: List[str] = None,
    format: str = None,
):
    try:
        dataset = _load_dataset(format, input, ignore)
    except ValueError as e:
        print(f"⚠️  Error: {str(e)}")
        return

    try:
        with model.open("rb") as fp:
            automl = AutoML.load(fp)
    except TypeError as e:
        print(f"⚠️  Error: {str(e)}")
        return

    print(f"⭐ Predicting {len(dataset)} items with the pipeline:")
    print(repr(automl.best_pipeline_))

    X = dataset.values
    y = automl.predict(X)


if __name__ == "__main__":
    app()
