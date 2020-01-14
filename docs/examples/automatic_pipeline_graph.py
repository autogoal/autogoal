from autogoal.kb import Word, Sentence, Matrix, Category, Vector, List, Algorithm, Union
from autogoal.kb import build_pipelines


class Vectorizer1:
    def run(self, input: Word()) -> Vector():
        pass


class Vectorizer2:
    def run(self, input: Word()) -> Vector():
        pass


class Tokenizer1:
    def run(self, input: Sentence()) -> List(Word()):
        pass


class Tokenizer2:
    def run(self, input: Sentence()) -> List(Word()):
        pass


class SentenceVectorizer:
    def __init__(
        self, 
        tokenizer: Algorithm(Sentence(), List(Word())), 
        vectorizer: Algorithm(Word(), Vector())
    ):
        pass

    def run(self, input: Sentence()) -> Matrix():
        pass


class Classifier1:
    def run(self, input: Union(Matrix(), Category())) -> Category():
        pass


class Classifier2:
    def run(self, input: Union(Matrix(), Category())) -> Category():
        pass


build_pipelines(input=Union(Sentence(), Category()), output=Category(), registry=[
    Vectorizer1, Vectorizer2, Tokenizer1, Tokenizer2, SentenceVectorizer, Classifier1, Classifier2
])
