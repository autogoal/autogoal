import fire


def main():
    from autogoal.datasets import pack, unpack, download

    fire.Fire(dict(
        pack=pack,
        unpack=unpack,
        download=download
    ), name='autogoal')


if __name__ == "__main__":
    main()
