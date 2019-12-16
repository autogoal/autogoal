# coding: utf8

"""Boilerplate code for unit testing, meant to be tested with `pytest`.
Here are some easy tests to begin with.
"""

from python_starter_pack import some_func
from python_starter_pack import say_hello


def test_some_func():
    assert some_func()


def test_say_hello():
    assert say_hello() is None


def test_the_meaning():
    assert 6 * 9 == int("42", base=13)

