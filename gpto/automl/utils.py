class InvalidPipeline(ValueError):
    """Raise when a pipeline is not valid after construction."""


def szip(*items):
    sizes = [len(i) for i in items]
    all_sizes = len(set(sizes)) == 1

    if not all_sizes:
        raise ValueError("All collections should be the same size (%s)." % str(sizes))

    return zip(*items)

def sdiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0


def draw(grammar, individual=None):
    import pydot

    graph = pydot.Dot()
    model = grammar.get_model()
