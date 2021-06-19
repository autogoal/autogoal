from autogoal.kb._semantics import Tensor
from autogoal.kb._semantics import Continuous, Dense, Discrete

Image = Tensor[2, Continuous, Dense]
ImageMask = Tensor[2, Discrete, Dense]