from autogoal.kb._semantics import Seq
from autogoal.grammar import CategoricalValue, DiscreteValue, ContinuousValue
from autogoal.experimental.augly._semantics import Image

from autogoal.utils import nice_repr

import augly.image import (
    brightness,
    change_aspect_ratio,
    contrast,
    crop,
    grayscale,
    opacity,
    pixelization,
    resize,
    rotate,
    saturation
)

from ._utils import AugLyTransformer

@nice_repr
class Brightness(AugLyTransformer):
    def __init__(
        self,
        factor: ContinuousValue(-10.0, 10.0)
        ):
        
        self.factor = factor
        super().__init__()

    def transform(self, X, y=None):
        return brightness(
            X, 
            self.factor)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class ChangeAspectRatio(AugLyTransformer):
    def __init__(
        self,
        ratio: ContinuousValue(0.1, 10.0)
        ):
        
        self.ratio = ratio
        super().__init__()

    def transform(self, X, y=None):
        return change_aspect_ratio(
            X, 
            self.ratio)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Contrast(AugLyTransformer):
    def __init__(
        self,
        factor: ContinuousValue(-10.0, 10.0)
        ):
        
        self.factor = factor
        super().__init__()

    def transform(self, X, y=None):
        return contrast(
            X, 
            self.factor)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Crop(AugLyTransformer):
    def __init__(
        self,
        x1: float = ContinuousValue(0, 1.0),
        y1: float = ContinuousValue(0, 1.0),
        x2: float = ContinuousValue(0, 1.0),
        y2: float = ContinuousValue(0, 1.0)
        ):
        
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        super().__init__()

    def transform(self, X, y=None):
        return crop(
            X, 
            self.x1,
            self.y1,
            self.x2,
            self.y2)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Grayscale(AugLyTransformer):
    def __init__(
        self,
        mode: CategoricalValue('luminosity', 'average')
        ):
        
        self.mode = mode
        super().__init__()

    def transform(self, X, y=None):
        return grayscale(
            X, 
            self.mode)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Opacity(AugLyTransformer):
    def __init__(
        self,
        level: ContinuousValue(0, 1.0)
        ):
        
        self.level = level
        super().__init__()

    def transform(self, X, y=None):
        return opacity(
            X, 
            self.level)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Pixelization(AugLyTransformer):
    def __init__(
        self,
        ratio: ContinuousValue(0.1, 1.0)
        ):
        
        self.ratio = ratio
        super().__init__()

    def transform(self, X, y=None):
        return pixelization(
            X, 
            self.ratio)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Resize(AugLyTransformer):
    def __init__(
        self,
        width: DiscreteValue(1, 5000),
        height: DiscreteValue(1, 5000)
        ):
        
        self.width = width
        self.height = height

        super().__init__()

    def transform(self, X, y=None):
        return resize(
            X, 
            self.width,
            self.height)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Rotate(AugLyTransformer):
    def __init__(
        self,
        degrees: ContinuousValue(0, 360.0)
        ):
        
        self.degrees = degrees
        super().__init__()

    def transform(self, X, y=None):
        return rotate(
            X, 
            self.degrees)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)

@nice_repr
class Saturation(AugLyTransformer):
    def __init__(
        self,
        factor: ContinuousValue(-10.0, 10.0)
        ):
        
        self.factor = factor
        super().__init__()

    def transform(self, X, y=None):
        return saturation(
            X, 
            self.factor)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Image) -> Image:
        return super().run(X)
