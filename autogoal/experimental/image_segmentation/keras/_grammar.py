from autogoal.contrib.keras._grammars import Module
from autogoal.contrib.keras._generated import MaxPooling2D, Activation
from autogoal.grammar import Path, GraphGrammar
from ._generated import Conv2DTranspose, Conv2D


class Modules:
    class ConvNN(Module):
        def make_top_level(self, top_level):
            if "PreprocessingModule" not in top_level:
                top_level.append("PreprocessingModule")

        def add_productions(self, grammar: GraphGrammar):
            grammar.add("PreprocessingModule", "DownBlock")
            grammar.add("DownBlock", Path("DoubleConv2DBlock", MaxPooling2D, "DownBlock"))
            grammar.add("DownBlock", Path("DoubleConv2DBlock", "UpBlock"))
            grammar.add("UpBlock", Path(Conv2DTranspose, "DoubleConv2DBlock", "UpBlock"))
            grammar.add("UpBlock", Path(Conv2DTranspose, "DoubleConv2DBlock"))
            grammar.add("DoubleConv2DBlock", Path(Conv2D, Conv2D))
