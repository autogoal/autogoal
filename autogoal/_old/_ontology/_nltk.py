# coding: utf8

import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

from .data import get_data_for, String


class _TokenizerBase:
    tokenizer = None

    def train(self, X, y=None):
        return self.run(X)

    def run(self, X):
        result = []

        for x in X:
            tokens = self.tokenizer.tokenize(x)
            x = String(x)
            x.tokens = tokens
            result.append(x)

        return result


class NLTKPunktTokenizer(_TokenizerBase):
    def __init__(self):
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    def __repr__(self):
        return "NLTKPunktTokenizer()"


class NLTKWordPunctTokenizer(_TokenizerBase):
    def __init__(self):
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()

    def __repr__(self):
        return "NLTKPunktTokenizer()"


class NLTKBigramCollocationFinder:
    def __init__(self):
        pass

    def train(self, X, y=None):
        return self.run(X)

    def run(self, X):
        finder = BigramCollocationFinder.from_documents(X)
        finder.nbest(BigramAssocMeasures.pmi, 10)

    def __repr__(self):
        return "NLTKBigramCollocation()"


class NLTKResolver:
    classes = {
        cls.__name__: cls
        for cls in [
            NLTKPunktTokenizer,
            NLTKWordPunctTokenizer,
            NLTKBigramCollocationFinder,
        ]
    }

    def resolve(self, instance, parameters):
        cls = self.classes[instance.name.split(".")[-1]]
        instance = cls(**parameters)
        return instance


def build_ontology_nltk(onto):
    Software = onto.Software
    NLTK = Software("NLTK")

    TokenizerClass = onto.Tokenizer

    PunktTokenizer = TokenizerClass("NLTKPunktTokenizer")
    PunktTokenizer.implementedIn = NLTK
    PunktTokenizer.importCode = "tokenizers/punkt/english.pickle"
    PunktTokenizer.hasInput = get_data_for(onto.DocumentCorpus)
    PunktTokenizer.hasOutput = get_data_for(onto.DocumentCorpus, onto.Tokenized)

    WordPunctTokenizer = TokenizerClass("NLTKWordPunctTokenizer")
    WordPunctTokenizer.implementedIn = NLTK
    WordPunctTokenizer.importCode = "nltk.tokenize.regexp.WordPunctTokenizer"
    WordPunctTokenizer.hasInput = get_data_for(onto.SentenceCorpus)
    WordPunctTokenizer.hasOutput = get_data_for(onto.SentenceCorpus, onto.Tokenized)

    ChunkingAlgorithmClass = onto["ChunkingAlgorithm"]

    CollocationsClass = onto["Collocations"]

    nltk_BigramCollocationFinder = CollocationsClass("NLTKBigramCollocationFinder")
    nltk_BigramCollocationFinder.implementedIn = NLTK
    nltk_BigramCollocationFinder.importCode = (
        "nltk.collocations.BigramCollocationFinder"
    )
    nltk_BigramCollocationFinder.hasInput = get_data_for(onto.WordCorpus)
    nltk_BigramCollocationFinder.hasOutput = get_data_for(onto.Paired)

    nltk_TrigramCollocationFinder = CollocationsClass("NLTKTrigramCollocationFinder")
    # nltk_TrigramCollocationFinder.implementedIn = NLTK
    nltk_TrigramCollocationFinder.importCode = (
        "nltk.collocations.TrigramCollocationFinder"
    )
    nltk_TrigramCollocationFinder.hasInput = get_data_for(onto.WordCorpus)
    nltk_TrigramCollocationFinder.hasOutput = get_data_for(onto.Paired)

    TaggingAlgorithmClass = onto["TaggingAlgorithmClass"]

    PartOfSpeechTaggingClass = onto["PartOfSpeechTagging"]

    nltk_word_tokenize = PartOfSpeechTaggingClass("NLTKWordTokenize")
    # nltk_word_tokenize.implementedIn = NLTK
    nltk_word_tokenize.importCode = "nltk.tokenize.word_tokenize"
    nltk_word_tokenize.hasInput = get_data_for(onto.Sentence, onto.Tokenized)
    nltk_word_tokenize.hasOutput = get_data_for(onto.Sentence, onto.PosTag)

    nltk_DefaultTagger = PartOfSpeechTaggingClass("NLTKDefaultTagger")
    # nltk_DefaultTagger.implementedIn = NLTK
    nltk_DefaultTagger.importCode = "nltk.tag.DefaultTagger"
    nltk_DefaultTagger.hasInput = get_data_for(onto.Sentence, onto.Tokenized)
    nltk_DefaultTagger.hasOutput = get_data_for(onto.Sentence, onto.PosTag)

    nltk_conlltags2tree = PartOfSpeechTaggingClass("NLTKChunkConlltags2tree")
    # nltk_conlltags2tree.implementedIn = NLTK
    nltk_conlltags2tree.importCode = "nltk.chunk.conlltags2tree"
    nltk_conlltags2tree.hasInput = get_data_for(onto.Sentence, onto.Tokenized)
    nltk_conlltags2tree.hasOutput = get_data_for(onto.Sentence, onto.PosTag)

