import textwrap

from autogoal.grammar import Discrete, generate_cfg
from autogoal.sampling import Sampler


def check_grammar(g, s):
    s = [si.strip() for si in s.split()]
    assert str(g).split() == s


def test_generate_from_class():
    class A:
        def __init__(self):
            pass

    check_grammar(generate_cfg(A), "<A> := A ()")


def test_generate_from_class_with_args():
    class A:
        def __init__(self, x: Discrete(1, 5)):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>   := A (x=<A_x>)
        <A_x> := discrete (min=1, max=5)
        """,
    )


def test_generate_from_method():
    def f():
        pass

    check_grammar(generate_cfg(f), "<f> := f ()")


def test_generate_from_method_with_args():
    def f(x: Discrete(1, 5)):
        pass

    check_grammar(
        generate_cfg(f),
        """
        <f>   := f (x=<f_x>)
        <f_x> := discrete (min=1, max=5)
        """,
    )


def test_sample_grammar():
    class A:
        def __repr__(self):
            return "A()"

    g = generate_cfg(A)
    assert str(g.sample()) == str(g())
