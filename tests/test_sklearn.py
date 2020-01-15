import pytest


def sklearn_classes():
    from autogoal.contrib.sklearn import _generated as sklearn
    import inspect

    classes = []

    for _, clss in inspect.getmembers(
        sklearn,
        lambda c: inspect.isclass(c)
        and issubclass(c, (sklearn.SklearnEstimator, sklearn.SklearnTransformer)),
    ):
        classes.append(clss)

    return classes


@pytest.mark.slow
@pytest.mark.parametrize("clss", sklearn_classes())
def test_create_grammar_for_generated_class(clss):
    from autogoal.grammar import generate_cfg
    generate_cfg(clss)


@pytest.mark.slow
@pytest.mark.parametrize("clss", sklearn_classes())
def test_sample_generated_class(clss):
    from autogoal.grammar import generate_cfg, Sampler

    grammar = generate_cfg(clss)
    sampler = Sampler(random_state=0)

    for _ in range(1000):
        grammar.sample(sampler=sampler)

