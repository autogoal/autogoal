import pytest

from autogoal.contrib import find_classes

classes = find_classes()


@pytest.mark.contrib
@pytest.mark.parametrize("clss", classes)
def test_create_grammar_for_generated_class(clss):
    from autogoal.grammar import generate_cfg
    generate_cfg(clss, registry=classes)


@pytest.mark.slow
@pytest.mark.contrib
@pytest.mark.parametrize("clss", classes)
def test_sample_generated_class(clss):
    from autogoal.grammar import generate_cfg, Sampler

    grammar = generate_cfg(clss, registry=classes)
    sampler = Sampler(random_state=0)

    for _ in range(1000):
        grammar.sample(sampler=sampler)
