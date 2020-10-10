import fire

from pathlib import Path
from autogoal import kb
from autogoal.contrib import find_classes


def demo():
    try:
        from streamlit.bootstrap import run
        run(Path(__file__).parent / "contrib" / "streamlit" / "demo.py", "", "")
    except ImportError:
        print("(!) Too run the demo you need streamlit installed.")
        print("(!) Fix it by running `pip install streamlit`.")


def graph(input:str, output:str):
    input_type = eval(input, kb.__dict__)

    print(input_type)


def main():
    from autogoal.datasets import pack, unpack, download

    fire.Fire(dict(
        pack=pack,
        unpack=unpack,
        download=download,
        demo=demo,
        graph=graph,
    ), name='autogoal')


if __name__ == "__main__":
    main()
