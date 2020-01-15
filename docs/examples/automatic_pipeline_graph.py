from autogoal.kb import Word, Sentence, Matrix, Category, Vector, List, algorithm, Union
from autogoal.kb import build_pipelines


class Vectorizer1:
    def run(self, input: Word()) -> Vector():
        pass


class Reductor1:
    def run(self, input: Matrix()) -> Matrix():
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
        tokenizer: algorithm(Sentence(), List(Word())),
        vectorizer: algorithm(Word(), Vector()),
    ):
        pass

    def run(self, input: Sentence()) -> Matrix():
        pass


class Classifier1:
    def run(self, input: Matrix()) -> Category():
        pass


class Classifier2:
    def run(self, input: Matrix()) -> Category():
        pass


space = build_pipelines(
    input=Sentence(),
    output=Category(),
    registry=[
        Vectorizer1,
        Vectorizer2,
        Tokenizer1,
        Tokenizer2,
        SentenceVectorizer,
        Classifier1,
        Classifier2,
        Reductor1,
    ],
)

path = space.sample()
print(path)
