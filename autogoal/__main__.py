import fire

from pathlib import Path


def demo():
    try:
        from streamlit.bootstrap import run
        run(Path(__file__).parent / "contrib" / "streamlit" / "demo.py", "", "")
    except ImportError:
        print("(!) Too run the demo you need streamlit installed.")
        print("(!) Fix it by running `pip install streamlit`.")


def main():
    from autogoal.datasets import pack, unpack, download

    fire.Fire(dict(
        pack=pack,
        unpack=unpack,
        download=download,
        demo=demo,
    ), name='autogoal')


if __name__ == "__main__":
    main()
