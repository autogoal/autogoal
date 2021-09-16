from autogoal.kb import AlgorithmBase

from typing import Tuple


class AuglyTransformer(AlgorithmBase):
    pass


def discrete_to_color(color: int) -> Tuple[int, int, int]:
    '''
    Convert a 8bit color `int` representation to `(r,g,b)` representation
    '''
    return (
        # Convert discrete value to a color Tuple using a mask
        color & 0xFF0000,
        color & 0x00FF00,
        color & 0x0000FF,
    )
