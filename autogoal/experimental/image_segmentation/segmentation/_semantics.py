from autogoal.kb._semantics import Tensor
from autogoal.kb._semantics import Continuous, Dense

Image = Tensor[2, Continuous, Dense]
ImageMask = Tensor[2, Continuous, Dense]