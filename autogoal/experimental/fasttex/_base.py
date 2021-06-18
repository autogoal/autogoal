from autogoal.kb import AlgorithmBase
import fasttext
import fasttext.util

from autogoal.grammar import CategoricalValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Word, Seq
from autogoal.kb import Supervised ,VectorCategorical, MatrixContinuousDense
from autogoal.utils import nice_repr

from os import remove
from time import time

class SupervisedTextClassifier(AlgorithmBase):
    """
    
    Assigns documents (such as emails, posts, text messages, product reviews, etc...) to one or multiple categories.
    
    It needs to be trained.
    
    All the labels start by the __label__ prefix, which is how fastText recognize what is a label or what is a word.
    
    If you want to assign multiple labels you can use Text_Descriptor 

    Params:
        
        -epoch: The number of times each examples is seen
        
        -lr:  How much the model changes after processing each example
        
        -wordNgrams: to refer to the concatenation any n consecutive tokens.
    """
    def __init__ (self,
                    epoch:DiscreteValue(min=5, max=25),
                    lr:ContinuousValue(0.1,1.0),
                    wordNgrams:DiscreteValue(1,3)
        ):
        self.epoch = epoch
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.model = None
        self._mode = "train"

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self,  X:Seq[Sentence], y:Supervised[VectorCategorical]) -> VectorCategorical :
        if self._mode == "train":
            return self._train(X,y)
        elif self._mode == "eval":
            return self._eval(X)

    def _train(self, X, y):
        with open('test.train','w') as g:
            text = [str(j) +" " +str(i) for i, j in zip(X,y) ]
            g.writelines(text)
            self.model = fasttext.train_supervised(input="test.train",epoch=self.epoch,lr=self.lr,wordNgrams=self.wordNgrams)
        remove("test.train")
        return y

    def _eval(self, X):
        res = []
        for text in X:
            text = text[:-1]
            res.append(self.model.predict(text)[0][0])
        return res


class Text_Descriptor:
    def __init__(self,*labels) -> None:
        self.labels =labels

    def __eq__(self, other):
        if isinstance(other,str):
            return other in self.labels
        if isinstance(other,Text_Descriptor):
            sorted(self.labels) == sorted(other.labels)
        return False
    def __str__(self):
        return " ".join(self.labels)


from re import compile as re_compile

class DataPreprocessing:
    """
    Apply some simple pre-processing to severals sentences.

    Params:
    - re: regular expression used to remove unwanted chars.
    - to_lower: if `True` text will be set to lowercase.
    """
    def __init__(self, re=r'[\?\!\,\.\:\;\{\}(\)\[\]\'\"]+', to_lower=True):
        self.re = re_compile(re)
        self.to_lower = to_lower

    def run(self, X: Seq[Sentence]):
        return [' '.join(self.re.split(s.lower() if self.to_lower else s)) for s in X]

class UnsupervisedWordRepresentation(AlgorithmBase):
    """
    Transform words to vectors. These vectors capture hidden information about a language, 
    like word analogies or semantic. 
    
    It uses a collection of documents to be trained.

    Params:

    - min_subword , max_subword: The subwords are all the substrings contained in a word 
    between the minimum size (`min_subword`) and the maximal size (`max_subword`).

    - dimension: controls the size of the vectors, the larger they are the more information 
    they can capture but requires more data to be learned.

    - model: The `skipgram` model learns to predict a target word thanks to a nearby word. 
    On the other hand, the `cbow` model predicts the target word according to its context.

    You can perform the transformation with `run(self, corpus, inputs)`.
    """
    def __init__ (self,
                    min_subword: DiscreteValue(min=1, max=3),
                    max_subword: DiscreteValue(min=3, max=6),
                    dimension: DiscreteValue(min=100, max=300),
                    model: CategoricalValue("skipgram", "cbow"),
                    epoch:DiscreteValue(min=5, max=25),
                    lr:ContinuousValue(0.1,1.0)
        ):
        self.min_subword = min_subword
        self.max_subword = max_subword
        self.dimension = dimension
        self.repr_model = model
        self.epoch = epoch
        self.lr = lr
        self.model = None

    def fit(self, X: Seq[Sentence], y=None):
        file = f'uwr_test_{time()}.test'
        with open(file, 'w') as f:
            f.writelines(X)

        self.model = fasttext.train_unsupervised(
                                file, self.repr_model,
                                minn=self.min_subword,
                                maxn=self.max_subword,
                                dim=self.dimension,
                                epoch=self.epoch,
                                lr=self.lr,
        )

        remove(file)
        return self

    def transform(self, _, y):
        return [self.model.get_word_vector(x) for x in y]

    def fit_transform(self, X: Seq[Sentence], y: Seq[Word]):
        self.fit(X, y)
        return self.transform(X, y)

    def run(self, corpus:Seq[Sentence], inputs:Seq[Word]) -> MatrixContinuousDense :
        self.fit_transform(corpus, inputs)

class UnsupervisedWordRepresentationPT(AlgorithmBase):
    """
    Version of class `UnsupervisedWordRepresentation` using pre-trained models.

    Params:

    - dimension: downloaded models always contains word vectors of size 300, you can resize
    them using this param. Default 300.

    - corpus: The source of pre-trained model, two options: `cc`(Common Crawl) or `wiki`(Wikipedia).
    Default `cc`.

    - lang: language code of the documents taken in the pre-trained model. Deafult `en`(English)

    You can perform the transformation with `run(self, inputs)`.

    In order to download the models, use class method: `download(lang)`.
    """
    def __init__ (self,
                    dimension: DiscreteValue(min=100, max=300)=300,
                    corpus: CategoricalValue('cc', 'wiki')='cc',
                    lang: CategoricalValue(
                                    'da', 'nl', 'en', 'fi', 'fr', 'de', 'hu',
                                    'it', 'no', 'pt', 'ru', 'es', 'sv', 'tr'
                                )='en',
        ):
        self.dimension = dimension
        self.corpus = corpus
        self.lang = lang
        self.model = None

    @classmethod
    def download(cls, lang='en'):
        try:
            fasttext.util.download_model(lang, if_exists='ignore')
        except Exception as ex:
            print(f"(!) [fasttext] Can't download pre-trained binary file with lang={lang}")
            raise ex

    def fit(self):
        if self.model is None:
            file = f"{self.corpus}.{self.lang}.300.bin"
            try:
                self.model = fasttext.load_model(file)
            except OSError:
                raise TypeError(
                    f"(!) [fasttext] Must download pre-trained binary file {file} first"
                )

            if self.model.get_dimension() > self.dimension:
                fasttext.util.reduce_model(self.model, self.dimension)

    def transform(self, X: Seq[Word]) -> MatrixContinuousDense :
        return [self.model.get_word_vector(x) for x in X]

    def fit_transform(self, X: Seq[Word]):
        self.fit()
        return self.transform(X)

    def run(self, inputs:Seq[Word]) -> MatrixContinuousDense :
        self.fit_transform(inputs)

class UnsupervisedWordAnalogies(AlgorithmBase):
    def __init__ (self,
                    min_subword: DiscreteValue(min=1, max=3),
                    max_subword: DiscreteValue(min=3, max=6),
                    dimension: DiscreteValue(min=100, max=300),
                    model: CategoricalValue("skipgram", "cbow"),
                    epoch:DiscreteValue(min=5, max=25),
                    lr:ContinuousValue(0.1,1.0)
        ):
        self.min_subword = min_subword
        self.max_subword = max_subword
        self.dimension = dimension
        self.repr_model = model
        self.epoch = epoch
        self.lr = lr
        self.model = None

    def fit(self, X: Seq[Sentence], y=None):
        file = f'uwr_test_{time()}.test'
        with open(file, 'w') as f:
            f.writelines(X)

        self.model = fasttext.train_unsupervised(
                                file, self.repr_model,
                                minn=self.min_subword,
                                maxn=self.max_subword,
                                dim=self.dimension,
                                epoch=self.epoch,
                                lr=self.lr,
        )

        remove(file)
        return self

    def transform(self, _, y):
        return [self.model.get_analogies(x) for x in y]

    def fit_transform(self, X: Seq[Sentence], y: Seq[Word]):
        self.fit(X, y)
        return self.transform(X, y)

    def run(self, corpus:Seq[Sentence], inputs:Seq[Word]) -> MatrixContinuousDense:
        self.fit_transform(corpus, inputs)
