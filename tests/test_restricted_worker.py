from autogoal.utils import RestrictedWorkerByJoin, Gb, Mb
import time
import numpy as np
import pytest


def func(wait_time, allocate_memory=0, raises=False):
    time.sleep(wait_time)
    memory_hog = {}

    for i in range(allocate_memory):
        memory_hog[i] = "Hello World"

    if raises:
        raise Exception("You asked for it.")


def test_no_restriction():
    fn = RestrictedWorkerByJoin(func, timeout=None, memory=None)
    fn(0, 1024)


def test_restrict_time():
    fn = RestrictedWorkerByJoin(func, timeout=1, memory=None)

    with pytest.raises(TimeoutError):
        fn(2, 1024)


def test_handles_exc():
    fn = RestrictedWorkerByJoin(func, timeout=None, memory=None)

    with pytest.raises(Exception) as e:
        fn(0, 0, True)

    assert str(e.value).startswith("You asked for it.")


def test_restrict_memory():
    fn = RestrictedWorkerByJoin(func, timeout=None, memory=1 * Gb)

    with pytest.raises(MemoryError):
        fn(0, 10 ** 8)


def test_restrict_memory_fails():
    fn = RestrictedWorkerByJoin(func, timeout=None, memory=1 * Mb)

    with pytest.raises(ValueError) as e:
        fn(0, 10 ** 8)

    assert isinstance(e.value, ValueError)
