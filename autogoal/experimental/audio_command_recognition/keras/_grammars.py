from autogoal.contrib.keras._grammars import Module
from autogoal.contrib.keras._generated import Dropout, Flatten
from autogoal.grammar import GraphGrammar, Path
from ._generated import Conv1D, MaxPooling1D


class Modules:
    class Conv1D(Module):
        def make_top_level(self, top_level):
            if "PreprocessingModule" not in top_level:
                top_level.append("PreprocessingModule")
            
        def add_productions(self, grammar: GraphGrammar):
            grammar.add("PreprocessingModule", Path("Conv1DModule", Flatten))

            grammar.add("Conv1DModule", Path("Conv1DModule", "Conv1DModule"))
            grammar.add("Conv1DModule", "Conv1DBlock")

            grammar.add("Conv1DBlock", Path("Conv1DCells", MaxPooling1D))
            grammar.add("Conv1DBlock", Path("Conv1DCells", MaxPooling1D, Dropout))

            grammar.add("Conv1DCells", Path("Conv1DCells", Conv1D))
            grammar.add("Conv1DCells", Conv1D)