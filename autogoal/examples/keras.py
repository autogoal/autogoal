# coding: utf8

from autogoal.ontology import build_nn_grammar


def main():
    grammar = build_nn_grammar()
    print(grammar)

    pipeline = grammar.sample()
    pipeline.compile(input_shape=(32,))
    print(pipeline.model.summary())


if __name__ == "__main__":
    main()
