from tensorflow.python.framework.ops import Tensor
from autogoal.kb._semantics import Continuous, Dense

Image = Tensor[2, Continuous, Dense]
