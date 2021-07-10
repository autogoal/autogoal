from autogoal.kb._semantics import Sentence, Seq
from autogoal.grammar import DiscreteValue

from autogoal.utils import nice_repr

from ._utils import BaseTransformer
from gramformer import Gramformer

@nice_repr
class GramCorrect(BaseTransformer):
    def __init__(
        self,
        max_candidates = DiscreteValue(1, 5),
        ):
        self.max_candidates = max_candidates
        super().__init__()

    def transform(self, X, y=None):
        self.gf = Gramformer(models = 1, use_gpu=False)
        _X = []
        for st in X:
            corrects = self.gf.correct(st, max_candidates=self.max_candidates)
            # gets only one candidate
            _X.append(corrects[0][0])
        return _X

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X) 

@nice_repr
class GramEdits(GramCorrect):
    def __init__(
        self, 
        max_candidates : DiscreteValue(1, 5)
        ):
        super().__init__(max_candidates=max_candidates)
    
    def transform(self, X, y):
        X_correct = super().transform(X, y=y)
        _X = []
        for infl, corr in zip(X, X_correct):
            edit = self.gf.get_edits(infl, corr)
            _X.append(edit)
        return _X

@nice_repr
class GramHighlighter(GramCorrect):
    def __init__(
        self, 
        max_candidates : DiscreteValue(1, 5)
        ):
        super().__init__(max_candidates=max_candidates)
    
    def transform(self, X, y):
        X_correct = super().transform(X, y=y)
        _X = []
        for infl, corr in zip(X, X_correct):
            highlight = self.gf.highlight(infl, corr)
            _X.append(highlight)
        return _X