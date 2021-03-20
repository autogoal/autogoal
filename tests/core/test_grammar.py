import pprint
from autogoal.sampling import Sampler

from autogoal.grammar import DiscreteValue, generate_cfg, Subset, CategoricalValue
from autogoal.kb import Document, Word, Stem, Seq, Sentence, algorithm
from autogoal.grammar import DiscreteValue, generate_cfg, Subset, CategoricalValue

class TextAlgorithm:
    def run(
        self, 
        input:Sentence()
    ) -> Document():
            pass

class StemWithDependanceAlgorithm:
    def __init__(
        self, 
        dependance:algorithm(Sentence(), Document())
    ):
        pass

    def run(
        self, 
        input:Word()
    ) -> Stem():
        pass

class StemAlgorithm:
    def run(
        self, 
        input:Word()
    ) -> Stem():
        pass

class HigherStemAlgorithm:
    def __init__(
        self, 
        dependance:algorithm(Word(), Stem())
    ):
        pass

    def run(
        self, 
        input:List(Word())
    ) -> List(Stem()):
        pass



def check_grammar(g, s):
    s = [si.strip() for si in s.split()]
    assert str(g).split() == s


def test_generate_from_registry_with_dependance():
    check_grammar(
        generate_cfg(HigherStemAlgorithm, registry=[StemAlgorithm, HigherStemAlgorithm, TextAlgorithm, StemWithDependanceAlgorithm] ), 
        """
        <HigherStemAlgorithm>               := HigherStemAlgorithm (dependance=<Algorithm[Word(), Stem()]>)
        <Algorithm[Word(), Stem()]>         := <StemAlgorithm> | <StemWithDependanceAlgorithm>
        <StemAlgorithm>                     := StemAlgorithm ()
        <StemWithDependanceAlgorithm>       := StemWithDependanceAlgorithm (dependance=<Algorithm[Sentence(), Document()]>)
        <Algorithm[Sentence(), Document()]> := <TextAlgorithm>
        <TextAlgorithm>                     := TextAlgorithm ()
        """)


def test_generate_from_class():
    class A:
        def __init__(self):
            pass

    check_grammar(generate_cfg(A), "<A> := A ()")


def test_generate_from_class_with_args():
    class A:
        def __init__(self, x: DiscreteValue(1, 5)):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>   := A (x=<A_x>)
        <A_x> := discrete (min=1, max=5)
        """,
    )

def test_subset_annotation_with_constants():
    class A:
        def __init__(self, features: Subset('Subset', "Hello", "World", 1)):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>      := A (features=<Subset>)
        <Subset> := { 'Hello' , 'World' , 1 }
        """,
    )

def test_subset_annotation_with_callables():
    class A:
        def __init__(self, features: Subset('Subset', DiscreteValue(1, 5), CategoricalValue('adam', 'sgd'))):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>      := A (features=<Subset>)
        <Subset> := { Discrete(min=1, max=5) , Categorical('adam', 'sgd') }
        """,
    )

def test_subset_annotation():
    class A:
        def __init__(self, features: Subset('Subset', DiscreteValue(1, 5), 'Hello', 1, None)):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>      := A (features=<Subset>)
        <Subset> := { Discrete(min=1, max=5) , 'Hello' , 1 , None }
        """,
    )

def test_subset_annotation():
    class A:
        def __init__(self, features: Subset('Subset', DiscreteValue(1, 5), 'Hello', 1, None)):
            pass

    check_grammar(
        generate_cfg(A),
        """
        <A>      := A (features=<Subset>)
        <Subset> := { Discrete(min=1, max=5) , 'Hello' , 1 , None }
        """,
    )

def test_sample_subset():
    class A:
        def __init__(self, features: Subset('Subset', DiscreteValue(1, 5), 'Hello', 1, None)):
            self.features = features

    g = generate_cfg(A)
    selected_features = g.sample().features
    selected = set([repr(feature) for feature in selected_features])
    assert selected.issubset([repr(feature) for feature in [DiscreteValue(1, 5), 'Hello', 1, None]])


def test_generate_from_method():
    def f():
        pass

    check_grammar(generate_cfg(f), "<f> := f ()")


def test_generate_from_method_with_args():
    def f(x: DiscreteValue(1, 5)):
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