#

from autogoal.grammar import generate_cfg


def test_generate_from_class():
    class A:
        def __init__(self):
            pass

    g = generate_cfg(A)
