from autogoal.kb import Word, Sentence, Matrix, Category, Vector, List, algorithm, Tuple
from autogoal.kb import build_pipelines
from autogoal.utils import nice_repr
from autogoal.grammar import Discrete


@nice_repr
class Vectorizer1:
    def run(self, input: Word()) -> Vector():
        pass


@nice_repr
class Reductor1:
    def run(self, input: Matrix()) -> Matrix():
        pass


@nice_repr
class Vectorizer2:
    def run(self, input: Word()) -> Vector():
        pass


@nice_repr
class Tokenizer1:
    def run(self, input: Sentence()) -> List(Word()):
        pass


@nice_repr
class Tokenizer2:
    def run(self, input: Sentence()) -> List(Word()):
        pass


@nice_repr
class SentenceVectorizer:
    def __init__(
        self,
        tokenizer: algorithm(Sentence(), List(Word())),
        vectorizer: algorithm(Word(), Vector()),
    ):
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer

    def run(self, input: Sentence()) -> Matrix():
        pass


class Classifier:
    def __init__(self):
        self.mode = "train"

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def run(self, input: Matrix()) -> Category():
        print(self.mode + "ing")


@nice_repr
class Classifier1(Classifier):
    def __init__(self, x: Discrete(1, 10)):
        self.x = x
        super().__init__()


@nice_repr
class Classifier2(Classifier):
    def __init__(self, y: Discrete(1, 10)):
        self.y = y
        super().__init__()


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


pipeline = space.sample()

pipeline.run([0,1,2])
pipeline.send("eval")
pipeline.run([2,3,4])
