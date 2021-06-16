from autogoal.contrib.keras._grammars import Module
from autogoal.contrib.keras._generated import Conv2D, MaxPooling2D, Activation
from autogoal.grammar import Path, GraphGrammar
from ._generated import Conv2DTranspose


class Modules:
    class ImageSegmenter(Module):
        def make_top_level(self, top_level: list):
            if "ImageSegmenter" not in top_level:
                top_level.append("ImageSegmenter")

        def add_productions(self, grammar: GraphGrammar):
            grammar.add("ImageSegmenter", "DownBlock")
            grammar.add("DownBlock", Path("DoubleConv2DBlock", MaxPooling2D, "DownBlock"))
            grammar.add("DownBlock", Path("DoubleConv2DBlock", "UpBlock"))
            grammar.add("UpBlock", Path(Conv2DTranspose, "DoubleConv2DBlock", "UpBlock"))
            grammar.add("UpBlock", Path(Conv2DTranspose, "DoubleConv2DBlock"))
            grammar.add("DoubleConv2DBlock", Path(Conv2D, Activation, Conv2D, Activation))
