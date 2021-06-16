from autogoal.contrib.keras._grammars import Module
from autogoal.grammar import Path, GraphGrammar
from ._generated import conv_2d_x, MaxPooling2D, conv_transpose_x, OutConv2D


class Modules:
    class UNET(Module):
        def add_productions(self, grammar: GraphGrammar):
            grammar.add("UNET", Path("UNETLayer16", OutConv2D))
            grammar.add("UNET", Path("UNETLayer32", OutConv2D))
            grammar.add("UNET", Path("UNETLayer64", OutConv2D))

            grammar.add("UNETLayer16", Path("UNETDown16", "UNETLayer32", "UNETUp16"))
            grammar.add("UNETLayer32", Path("UNETDown32", "UNETLayer64", "UNETUp32"))
            grammar.add("UNETLayer64", Path("UNETDown64", "UNETLayer128", "UNETUp64"))
            grammar.add("UNETLayer128", Path("UNETDown128", "UNETLayer256", "UNETUp128"))
            grammar.add("UNETLayer256", Path("UNETDown256", "UNETLayer512", "UNETUp256"))

            grammar.add("UNETLayer128", Path("UNETDown128", "UNETBottleNeck128", "UNETUp128"))
            grammar.add("UNETLayer256", Path("UNETDown256", "UNETBottleNeck256", "UNETUp256"))
            grammar.add("UNETLayer512", Path("UNETDown512", "UNETBottleNeck512", "UNETUp512"))

            grammar.add("UNETDown16", Path(conv_2d_x(16), conv_2d_x(16), MaxPooling2D))
            grammar.add("UNETDown32", Path(conv_2d_x(32), conv_2d_x(32), MaxPooling2D))
            grammar.add("UNETDown64", Path(conv_2d_x(64), conv_2d_x(64), MaxPooling2D))
            grammar.add("UNETDown128", Path(conv_2d_x(128), conv_2d_x(128), MaxPooling2D))
            grammar.add("UNETDown256", Path(conv_2d_x(256), conv_2d_x(256), MaxPooling2D))
            grammar.add("UNETDown512", Path(conv_2d_x(512), conv_2d_x(512), MaxPooling2D))

            grammar.add("UNETBottleNeck128", Path(conv_2d_x(128), conv_2d_x(128)))
            grammar.add("UNETBottleNeck256", Path(conv_2d_x(256), conv_2d_x(256)))
            grammar.add("UNETBottleNeck512", Path(conv_2d_x(512), conv_2d_x(512)))

            grammar.add("UNETUp16", Path(conv_transpose_x(16), conv_2d_x(16)))
            grammar.add("UNETUp32", Path(conv_transpose_x(32), conv_2d_x(32)))
            grammar.add("UNETUp64", Path(conv_transpose_x(64), conv_2d_x(64)))
            grammar.add("UNETUp128", Path(conv_transpose_x(128), conv_2d_x(128)))
            grammar.add("UNETUp256", Path(conv_transpose_x(256), conv_2d_x(256)))
            grammar.add("UNETUp512", Path(conv_transpose_x(512), conv_2d_x(512)))
