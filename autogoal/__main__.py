import collections
import inspect
import logging
from os import stat
from pathlib import Path
from typing import List

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from autogoal.contrib import (
    find_classes,
    status,
    ContribStatus,
    download as download_contrib,
)
from autogoal.kb import VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.utils import Gb, Min
from autogoal.datasets import datapath, get_datasets_list, download, dummy
import autogoal.logging

autogoal.logging.setup("WARNING")

logger = autogoal.logging.logger()
console = Console()


app = typer.Typer(name="autogoal")
contrib_app = typer.Typer(name="contrib")
automl_app = typer.Typer(name="ml")
data_app = typer.Typer(name="data")


app.add_typer(contrib_app)
app.add_typer(automl_app)
app.add_typer(data_app)


@app.callback()
def main():
    """
    🤩 Manage AutoGOAL directly from the CLI.
    """


@app.command()
def demo():
    """
    🌟 Launch streamlit demo.
    """

    try:
        from streamlit.bootstrap import run

        run(Path(__file__).parent / "contrib" / "streamlit" / "demo.py", "", "")
    except ImportError:
        console.print("(!) Too run the demo you need streamlit installed.")
        console.print("(!) Fix it by running `pip install autogoal[streamlit]`.")


@contrib_app.callback()
def contrib_main():
    """
    🔍 Inspect contrib libraries and algorithms.
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
    ⚙️ List all currently available contrib algorithms.
    """
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


@contrib_app.command("status")
def contrib_status():
    """
    ✔️ Shows the status of all contrib libraries.
    """
    table = Table("🛠️  Contrib", "✔️ Status")

    statuses = {
        ContribStatus.RequiresDependency: "🔴 Required dependency",
        ContribStatus.RequiresDownload: "🔴 Requires download",
        ContribStatus.Ready: "🟢 Ready",
    }

    for key, value in status().items():
        table.add_row(key, statuses[value])

    console.print(table)


@contrib_app.command("download")
def contrib_download(
    contrib=typer.Argument(
        ..., help="Name of the contrib, e.g., `sklearn` or `nltk`, or `all`."
    )
):
    """
    💾 Download necessary contrib files.
    """
    if status()[f"autogoal.contrib.{contrib}"] == ContribStatus.Ready:
        console.print(f"✅ Nothing to download for contrib `{contrib}`.")
    elif download_contrib(contrib):
        console.print(f"✅ Succesfully downloaded files for contrib `{contrib}`.")
    else:
        console.print(f"❌ Cannot download files for contrib `{contrib}`.")


@automl_app.callback()
def automl_callback():
    """
    🤖 Fit and predict with an AutoML model.
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
        columns_to_ignore = [dataset.columns[i] for i in ignore]
        dataset = dataset.drop(columns=columns_to_ignore)

    return dataset


@automl_app.command("fit")
def automl_fit(
    input: Path,
    output: Path = Path("automl.bin"),
    target: str = None,
    ignore_cols: List[int] = typer.Option([]),
    evaluation_timeout: int = 5 * Min,
    memory_limit: int = 4 * Gb,
    search_timeout: int = 60 * 60,
    pop_size: int = 20,
    iterations: int = 100,
    random_state: int = None,
    format: str = None,
):
    """
    🏃 Train an AutoML instance on a dataset.
    """

    try:
        dataset = _load_dataset(format, input, ignore_cols)
    except ValueError as e:
        logger.error(f"⚠️  Error: {str(e)}")
        return

    if target is None:
        target = dataset.columns[-1]

    columns = [c for c in dataset.columns if c != target]

    X = dataset[columns].values
    y = dataset[target].values

    automl = AutoML(
        output=VectorCategorical(),
        search_kwargs=dict(
            evaluation_timeout=evaluation_timeout,
            memory_limit=memory_limit,
            search_timeout=search_timeout,
            pop_size=pop_size,
        ),
        random_state=random_state,
        search_iterations=iterations,
    )

    console.print(f"🏃 Training on {len(dataset)} items.")
    automl.fit(X, y, logger=RichLogger())

    with output.open("wb") as fp:
        automl.save(fp)

    console.print(f"💾 Saving model to [green]{output.absolute()}[/].")


@automl_app.command("predict")
def automl_predict(
    input: Path,
    output: Path = Path("output.csv"),
    model: Path = Path("automl.bin"),
    ignore_cols: List[int] = typer.Option([]),
    format: str = None,
):
    """
    🔮 Predict with a previously trained AutoML instance.
    """

    try:
        dataset = _load_dataset(format, input, ignore_cols)
    except ValueError as e:
        logger.error(f"⚠️  Error: {str(e)}")
        return

    try:
        with model.open("rb") as fp:
            automl = AutoML.load(fp)
    except TypeError as e:
        logger.error(f"⚠️  Error: {str(e)}")
        return

    console.print(f"🔮 Predicting {len(dataset)} items with the pipeline:")
    console.print(repr(automl.best_pipeline_))

    X = dataset.values
    y = automl.predict(X)

    with output.open("wt") as fp:
        df = pd.DataFrame(y, columns=["y"])
        df.to_csv(fp)

    console.print(f"💾 Predictions saved to [blue]{output.absolute()}[/]")


@automl_app.command("inspect")
def automl_inspect(model: Path = Path("automl.bin")):
    """
    🔍 Inspect a trained AutoML model.
    """

    with model.open("rb") as fp:
        automl = AutoML.load(fp)

    console.print(f"🔍 Inspecting AutoML model: [green]{model.absolute()}[/]")

    console.print(f"⭐ Best pipeline (score={automl.best_score_:0.3f}):")
    console.print(repr(automl.best_pipeline_))


@data_app.callback()
def data_callback():
    """
    📚 Download, inspect, and generate training data.
    """


@data_app.command("list")
def data_list():
    """
    🔍 List the available datasets.
    """

    datasets = get_datasets_list()

    table = Table("📚 Dataset", "💾", "🔗 URL")

    for item, url in sorted(datasets.items(), key=lambda t: t[0]):
        path = datapath(item)

        if path.exists():
            table.add_row(item, "✔️", url)
        else:
            table.add_row(item, "", url)

    console.print(table)


@data_app.command("download")
def data_download(
    datasets: List[str] = typer.Argument(
        ..., help="Name of one or more specific datasets to download, or 'all'."
    )
):
    """
    ⏬ Download a dataset.

    Pass a name to directly download that dataset.
    Otherwise, this command will show an interactive menu.
    """

    if "all" in datasets:
        datasets = get_datasets_list().keys()

    for dataset in datasets:
        download(dataset)


@data_app.command("gen")
def data_generate():
    """
    🎲 Generate a random dataset.
    """


if __name__ == "__main__":
    try:
        app(prog_name="autogoal")
    except Exception as e:
        console.print(f'⚠️  The command failed with message:\n"{str(e)}".')

        if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
            logger.exception("Check the traceback below.")
