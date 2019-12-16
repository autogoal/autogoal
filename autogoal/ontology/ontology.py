# coding: utf8

from owlready2 import *
from pathlib import Path
import enlighten


ontology_path = Path(__file__).parent.parent / "docs" / "ontoml.owl"
onto_path.append(str(ontology_path.parent))

try:
    onto = get_ontology("https://knowledge-learning.github.io/ontoml/ontoml.owl").load()
except:
    onto = get_ontology("https://knowledge-learning.github.io/ontoml/ontoml.owl")


SWRL_rules = [
    # Defining the input and output of a CompositePipeline
    "hasFirstStep(?P,?A) ^ hasInput(?A,?D) -> hasInput(?P,?D)",
    "hasNextStep(?P,?B) ^ hasOutput(?B,?D) -> hasOutput(?P,?D)",

    # Restricting the steps' input/output to match
    "hasFirstStep(?P,?A) ^ hasOutput(?A,?D) ^ hasNextStep(?P,?B) -> hasInput(?B,?D)",

    # Restricting which algorithms can be connected
    "hasOutput(?A, ?D1) ^ hasInput(?B, ?D2) ^ isCoercibleTo(?D1, ?D2) -> canConnect(?A, ?B)",
]


with onto:
    ### Data types

    class Data(Thing):
        pass

    class isCoercibleTo(Data >> Data):
        pass

    class isCoercibleFrom(Data >> Data):
        inverse_property = isCoercibleTo

    #### Concrete data types

    class Sequence(Data):
        pass

    class Categorical(Data):
        pass

    class Numeric(Data):
        pass

    class Matrix(Numeric):
        pass

    class Vector(Numeric, Sequence):
        pass

    class Discrete(Numeric):
        pass

    class Continuous(Numeric):
        pass

    class Sparse(Numeric):
        pass

    class Dense(Numeric):
        pass

    class Text(Data):
        pass

    class NaturalText(Text):
        pass

    class DocumentCorpus(Sequence, NaturalText):
        pass

    class SentenceCorpus(Sequence, NaturalText):
        pass

    class Sentence(NaturalText):
        pass

    class WordCorpus(Sequence, NaturalText):
        pass

    class Structured(Data):
        pass

    class Mapping(Structured):
        pass

    class Paired(Structured):
        pass

    class Semantic(Data):
        pass

    class Entities(Sequence, Semantic):
        pass

    class PosTag(Sequence, Semantic):
        pass

    class Tokenized(Semantic):
        pass

    ### Basic pipeline structure

    class Pipeline(Thing):
        pass

    class hasInput(FunctionalProperty, Pipeline >> Data):
        pass

    class hasOutput(FunctionalProperty, Pipeline >> Data):
        pass

    class Algorithm(Pipeline):
        pass

    class Adaptor(Algorithm):
        pass

    class canConnect(Algorithm >> Algorithm):
        pass

    class CompositePipeline(Pipeline):
        pass

    class SequentialPipeline(CompositePipeline):
        pass

    class hasStartStep(FunctionalProperty, SequentialPipeline >> Algorithm):
        pass

    class hasEndStep(FunctionalProperty, SequentialPipeline >> Pipeline):
        pass

    class ParallelPipeline(CompositePipeline):
        pass

    class hasFirstStep(FunctionalProperty, ParallelPipeline >> Pipeline):
        pass

    class hasSecondStep(FunctionalProperty, ParallelPipeline >> Pipeline):
        pass

    ### Machine Learning Algorithms

    class Estimator(Algorithm):
        pass

    class Supervised(Estimator):
        pass

    class Classifier(Supervised):
        pass

    class Regressor(Supervised):
        pass

    class Unsupervised(Estimator):
        pass

    class Transformer(Algorithm):
        pass

    class Clusterer(Unsupervised):
        pass

    ### Text Processing Algorithms

    class TextAlgorithm(Algorithm):
        pass

    class Tokenizer(TextAlgorithm):
        pass

    class SimilarityAlgorithm(Algorithm):
        pass

    class Concordance(SimilarityAlgorithm):
        pass

    class SimilarWords(SimilarityAlgorithm):
        pass

    class CommonContext(SimilarityAlgorithm):
        pass

    class CountingAlgorithm(TextAlgorithm):
        pass

    class CountCharacters(CountingAlgorithm):
        pass

    class CountTokens(CountingAlgorithm):
        pass

    class TokensLength(CountingAlgorithm):
        pass

    class FrequencyMap(CountingAlgorithm):
        pass

    class NormalizationAlgorithm(TextAlgorithm):
        pass

    class RemovePunctuation(NormalizationAlgorithm):
        pass

    class RemoveStopwords(NormalizationAlgorithm):
        pass

    class ChunkingAlgorithm(TextAlgorithm):
        pass

    class Collocations(ChunkingAlgorithm):
        pass

    class TaggingAlgorithm(TextAlgorithm):
        pass

    class PartOfSpeechTagging(TaggingAlgorithm):
        pass

    ### Implementation details

    class Software(Thing):
        pass

    class implementedIn(FunctionalProperty, Algorithm >> Software):
        pass

    class importCode(FunctionalProperty, Algorithm >> str):
        pass

    ### Parameters

    class HyperParameter(Thing):
        pass

    class hasParameter(Algorithm >> HyperParameter):
        pass

    class ContinuousHyperParameter(HyperParameter):
        pass

    class hasMinFloatValue(FunctionalProperty, ContinuousHyperParameter >> float):
        pass

    class hasMaxFloatValue(FunctionalProperty, ContinuousHyperParameter >> float):
        pass

    class hasDefaultFloatValue(FunctionalProperty, ContinuousHyperParameter >> float):
        pass

    class DiscreteHyperParameter(HyperParameter):
        pass

    class hasMinIntValue(FunctionalProperty, DiscreteHyperParameter >> int):
        pass

    class hasMaxIntValue(FunctionalProperty, DiscreteHyperParameter >> int):
        pass

    class hasDefaultIntValue(FunctionalProperty, DiscreteHyperParameter >> int):
        pass

    class BooleanHyperParameter(HyperParameter):
        pass

    class hasDefaultBoolValue(FunctionalProperty, BooleanHyperParameter >> bool):
        pass

    class StringHyperParameter(HyperParameter):
        pass

    class hasStringValues(StringHyperParameter >> str):
        pass

    class hasDefaultStringValue(FunctionalProperty, StringHyperParameter >> str):
        pass


def save_ontology():
    solve_coercible()
    solve_can_connect()

    path = str(Path(__file__).parent.parent / "docs" / "ontoml.owl")
    onto.save(file=path)


def solve_coercible():
    for i in Data.instances():
        for j in Data.instances():
            if i == j:
                i.isCoercibleTo.append(j)
                continue

            i_bases = set(i.is_a)
            j_bases = set(j.is_a)

            if _can_coerce(i_bases, j_bases):
                i.isCoercibleTo.append(j)
                print(i, '--> isCoercibleTo -->', j)


def _can_coerce(i_bases, j_bases):
    for cls in j_bases:
        if not _has_children_in(cls, i_bases):
            return False

    return True


def _has_children_in(cls, classes):
    if isinstance(cls, Not):
        return all(not issubclass(cli, cls.Class) for cli in classes)

    return any(issubclass(cli, cls) for cli in classes)


def solve_can_connect():
    instances = list(Algorithm.instances())

    manager = enlighten.get_manager()
    counter = manager.counter(total=len(instances) * len(instances), unit='pairs')

    for i in Algorithm.instances():
        if i.hasOutput is None:
            counter.update(len(instances))
            continue

        for j in Algorithm.instances():
            counter.update()

            if i == j or j.hasInput is None:
                continue

            if j.hasInput in i.hasOutput.isCoercibleTo:
                i.canConnect.append(j)
                print(i, '--> canConnect -->', j)

    counter.close()
    manager.stop()


if __name__ == "__main__":
    from ._sklearn import build_ontology_sklearn
    from ._nltk import build_ontology_nltk
    from ._adapters import build_ontology_adapters

    import sys

    if ontology_path.exists():
        print("(!) Delete %s first..." % ontology_path)
        sys.exit(127)

    if 'nltk' in sys.argv or 'full' in sys.argv:
        build_ontology_nltk(onto)

    if 'sklearn' in sys.argv or 'full' in sys.argv:
        build_ontology_sklearn(onto)

    if 'ontoml' in sys.argv or 'full' in sys.argv:
        build_ontology_adapters(onto)

    save_ontology()

