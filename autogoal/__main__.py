import typer
import collections

from pathlib import Path
from autogoal import kb
from autogoal.contrib import find_classes


app = typer.Typer(name="AutoGOAL")
contrib_app = typer.Typer(name="contrib")

app.add_typer(contrib_app)


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
def contrib_list(verbose: bool = False, include: str = None, exclude: str = None):
    """
    List all currently available contrib algorithms.
    """
    from autogoal.contrib import find_classes

    classes = find_classes(include=include, exclude=exclude)
    classes_by_contrib = collections.defaultdict(list)

    for cls in classes:
        classes_by_contrib[str(cls).split(".")[2]].append(cls)

    typer.echo(
        f"⚙️  Found a total of {len(classes)} matching algorithms.", color="blue"
    )

    for contrib, clss in classes_by_contrib.items():
        typer.echo(f"🛠️  {contrib}: {len(clss)} algorithms.")

        if verbose:
            for cls in clss:
                typer.echo(f" 🔹 {cls.__name__}")


if __name__ == "__main__":
    app()
