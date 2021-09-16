from augly.image.transforms import BaseTransform
import augly.image.transforms as transforms
import PIL

from autogoal.grammar import (
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
    BooleanValue,
)

from typing import Tuple


from autogoal.experimental.augly_tony.semantic import Image
from _util import AuglyTransformer, discrete_to_color


class AugLyImageTransformer(AuglyTransformer):
    """
    Base class for augLy image transformers
    """

    def __init__(self):
        super().__init__()
        self._transformer: BaseTransform = None

    def get_transformer(self) -> BaseTransform:
        pass

    def run(self, X: Image) -> Image:
        if self._transformer is None:
            self._transformer = self.get_transformer()

        return self._transformer(Image)


class BlurTransformer(AugLyImageTransformer):
    """
    Blurs the image
    """

    def __init__(
        self,
        radius: ContinuousValue(0, 10),
    ):
        super().__init__()
        self.radius = radius

    def get_transformer(self) -> BaseTransform:
        return transforms.Blur(radius=self.radius)


class BrightnessTransformer(AugLyImageTransformer):
    """
    Alters the brightness of the image
    """

    def __init__(
        self,
        factor: ContinuousValue(-10, 10),
    ):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Brightness(factor=self.factor)


class ChangeAspectRatioTransformer(AugLyImageTransformer):
    """
    Alters the aspect ratio of the image
    """

    def __init__(
        self,
        ratio: ContinuousValue(0.1, 10),
    ):
        super().__init__()
        self.ratio = ratio

    def get_transformer(self) -> BaseTransform:
        return transforms.ChangeAspectRatio(ratio=self.ratio)


class ClipImageSizeTransformer(AugLyImageTransformer):
    """
    Scales the image up or down if necessary to fit in the given min and max
    resolution
    """

    def __init__(
        self,
        min_resolution: DiscreteValue(100, 200),
        max_resolution: DiscreteValue(200, 1000),
    ):
        super().__init__()
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def get_transformer(self) -> BaseTransform:
        return transforms.ClipImageSize(
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
        )


class ColorJitterTransformer(AugLyImageTransformer):
    """
    Color jitters the image
    """

    def __init__(
        self,
        brightness_factor: ContinuousValue(-5, 5),
        contrast_factor: ContinuousValue(-5, 5),
        saturation_factor: ContinuousValue(-5, 5),
    ):
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor

    def get_transformer(self) -> BaseTransform:
        return transforms.ColorJitter(
            brightness_factor=self.brightness_factor,
            contrast_factor=self.contrast_factor,
            saturation_factor=self.saturation_factor,
        )


class ContrastTransformer(AugLyImageTransformer):
    """
    Alters the contrast of the image
    """

    def __init__(self, factor: ContinuousValue(-5, 5)):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Contrast(factor=self.factor)


class ConvertColorTransformer(AugLyImageTransformer):
    """
    The mode of an image is a string which defines the type and depth of a pixel in the image. Each pixel uses the full range of the bit depth. So a 1-bit pixel has a range of 0-1, an 8-bit pixel has a range of 0-255 and so on.

    `1` (1-bit pixels, black and white, stored with one pixel per byte)

    `L` (8-bit pixels, black and white)

    `RGB` (3x8-bit pixels, true color)
    """

    def __init__(
        self,
        mode: CategoricalValue("1", "L", "RGB"),
    ):
        super().__init__()
        self.mode = mode

    def get_transformer(self) -> BaseTransform:
        return transforms.ConvertColor(mode=self.mode)


class CropTransformer(AugLyImageTransformer):
    """
    Crops the image
    """

    def __init__(
        self,
        x1: float = ContinuousValue(0, 1),
        y1: float = ContinuousValue(0, 1),
        x2: float = ContinuousValue(0, 1),
        y2: float = ContinuousValue(0, 1),
    ):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_transformer(self) -> BaseTransform:
        return transforms.Crop(
            x1=self.x1,
            x2=self.x2,
            y1=self.y1,
            y2=self.y2,
        )


class EncodingQualityTransformer(AugLyImageTransformer):
    """
    Changes the JPEG encoding quality level
    """

    def __init__(self, quality: DiscreteValue(0, 100)):
        self.quality = quality
        super().__init__()

    def get_transformer(self) -> BaseTransform:
        return transforms.EncodingQuality(quality=self.quality)


class GrayscaleTransformer(AugLyImageTransformer):
    """
    Alters an image to be grayscale
    """

    def __init__(
        self,
        mode: CategoricalValue("luminosity", "average"),
    ):
        self.mode = mode
        super().__init__()

    def get_transformer(self) -> BaseTransform:
        return transforms.Grayscale(mode=self.mode)


class HFlipTransformer(AugLyImageTransformer):
    """
    Horizontally flips an image
    """

    def get_transformer(self) -> BaseTransform:
        return transforms.HFlip()


class VFlipTransformer(AugLyImageTransformer):
    """
    Vertically flips an image
    """

    def get_transformer(self) -> BaseTransform:
        return transforms.HFlip()


# TODO: Improve with addional grammar parameters
class MemeFormatTransformer(AugLyImageTransformer):
    """
    Creates a new image that looks like a meme, given text and an image
    """

    def __init__(self):
        super().__init__()

    def get_transformer(self) -> BaseTransform:
        return transforms.MemeFormat()


class OpacityTransformer(AugLyImageTransformer):
    """
    Alters the opacity of an image
    """

    def __init__(
        self,
        level: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.level = level

    def get_transformer(self) -> BaseTransform:
        return transforms.Opacity(level=self.level)


class OverlayEmojiTransformer(AugLyImageTransformer):
    """
    Overlay an emoji onto the original image
    """

    def __init__(
        self,
        opacity: ContinuousValue(0, 1),
        emoji_size: ContinuousValue(0, 1),
        x_pos: ContinuousValue(0, 1),
        y_pos: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def get_transformer(self) -> BaseTransform:
        return transforms.OverlayEmoji(
            opacity=self.opacity,
            emoji_size=self.emoji_size,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
        )


class OverlayOntoScreenshotTransformer(AugLyImageTransformer):
    """
    Overlay the image onto a screenshot template so it looks like it was
    screenshotted on Instagram
    """

    def __init__(
        self,
        crop_src_to_fit: BooleanValue(),
        resize_src_to_match_template: BooleanValue(),
    ):
        super().__init__()
        self.crop_src_to_fit = crop_src_to_fit
        self.resize_src_to_match_template = resize_src_to_match_template

    def get_transformer(self) -> BaseTransform:
        return transforms.OverlayOntoScreenshot(
            crop_src_to_fit=self.crop_src_to_fit,
            resize_src_to_match_template=self.resize_src_to_match_template,
        )


class OverlayStripesTransformer(AugLyImageTransformer):
    """
    Overlay stripe pattern onto the image (by default, stripes are horizontal)
    """

    def __init__(
        self,
        line_width: ContinuousValue(0, 1),
        line_color: DiscreteValue(0, 0xFFFFFF),
        line_angle: ContinuousValue(0, 360),
        line_density: ContinuousValue(0, 1),
        line_type: CategoricalValue("dotted", "dashed", "solid"),
        line_opacity: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.line_width = line_width
        self.line_color = line_color
        self.line_angle = line_angle
        self.line_density = line_density
        self.line_type = line_type
        self.line_opacity = line_opacity

    def get_transformer(self) -> BaseTransform:
        return transforms.OverlayStripes(
            line_width=self.line_width,
            line_color=discrete_to_color(self.line_color),
            line_angle=self.line_angle,
            line_density=self.line_density,
            line_type=self.line_type,
            line_opacity=self.line_opacity,
        )


# TODO: Improve by adding diferent text inputs
class OverlayTextTransformer(AugLyImageTransformer):
    """
    Overlay text onto the image (by default, text is randomly overlaid)
    """

    def __init__(
        self,
        font_size: ContinuousValue(0, 1),
        opacity: ContinuousValue(0, 1),
        color: DiscreteValue(0, 0xFFFFFF),
        x_pos: ContinuousValue(0, 1),
        y_pos: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.font_size = font_size
        self.opacity = opacity
        self.color = (color,)
        self.x_pos = x_pos
        self.y_pos = y_pos

    def get_transformer(self) -> BaseTransform:
        return transforms.OverlayText(
            font_size=self.font_size,
            opacity=self.opacity,
            color=discrete_to_color(self.color),
            x_pos=self.x_pos,
            y_pos=self.y_pos,
        )


class PadTransformer(AugLyImageTransformer):
    """
    Pads the image and add a border
    """

    def __init__(
        self,
        w_factor: ContinuousValue(0, 1),
        h_factor: ContinuousValue(0, 1),
        color: DiscreteValue(0, 0xFFFFFF),
    ):
        super().__init__()
        self.w_factor = w_factor
        self.h_factor = h_factor
        self.color = color

    def get_transformer(self) -> BaseTransform:
        return transforms.Pad(
            w_factor=self.w_factor,
            h_factor=self.h_factor,
            color=discrete_to_color(self.color),
        )


class PadSquareTransformer(AugLyImageTransformer):
    """
    Pads the shorter edge of the image such that it is now square-shaped
    """

    def __init__(self, color: ContinuousValue(0, 0xFFFFFF)):
        super().__init__()
        self.color = color

    def get_transformer(self) -> BaseTransform:
        return transforms.PadSquare(color=self.color)


class PerspectiveTransformTransformer(AugLyImageTransformer):
    """
    Apply a perspective transform to the image so it looks like it was taken
    as a photo from another device.
    """

    def __init__(
        self,
        sigma: ContinuousValue(0, 1000),
        dx: ContinuousValue(0, 1),
        dy: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.sigma = sigma
        self.dx = dx
        self.dy = dy

    def get_transformer(self) -> BaseTransform:
        return transforms.PerspectiveTransform(
            sigma=self.sigma,
            dx=self.dx,
            dy=self.dy,
        )


class PixelizationTransformer(AugLyImageTransformer):
    """
    Pixelizes an image
    """

    def __init__(self, ratio: ContinuousValue(0, 10)):
        super().__init__()
        self.ratio = ratio

    def get_transformer(self) -> BaseTransform:
        return transforms.Pixelization(ratio=self.ratio)


class RandomNoiseTransformer(AugLyImageTransformer):
    """
    Adds random noise to the image
    """

    def __init__(
        self,
        mean: ContinuousValue(0, 10),
        var: ContinuousValue(0, 10),
    ):
        super().__init__()
        self.mean = mean
        self.var = var

    def get_transformer(self) -> BaseTransform:
        return transforms.RandomNoise(mean=self.mean, var=self.var)


class ResizeTransformer(AugLyImageTransformer):
    """
    Resizes an image
    """

    def __init__(
        self,
        width: DiscreteValue(10, 1000),
        height: DiscreteValue(10, 1000),
    ):
        super().__init__()
        self.width = width
        self.height = height

    def get_transformer(self) -> BaseTransform:
        return transforms.Resize(
            width=self.width,
            height=self.height,
        )


class RotateTransformer(AugLyImageTransformer):
    """
    Rotates the image
    """

    def __init__(self, degrees: ContinuousValue(0, 360)):
        super().__init__()
        self.degrees = degrees

    def get_transformer(self) -> BaseTransform:
        return transforms.Rotate(degrees=self.degrees)


class SaturationTransformer(AugLyImageTransformer):
    """
    Alters the saturation of an image
    """

    def __init__(
        self,
        factor: ContinuousValue(-10, 10),
    ):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Saturation(factor=self.factor)


class ScaleTransformer(AugLyImageTransformer):
    """
    Alters the resolution of an image
    """

    def __init__(
        self,
        factor: ContinuousValue(-10, 10),
        interpolation: CategoricalValue(
            PIL.Image.NEAREST,
            PIL.Image.BOX,
            PIL.Image.BILINEAR,
            PIL.Image.HAMMING,
            PIL.Image.BICUBIC,
            PIL.Image.LANCZOS,
        ),
    ):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation

    def get_transformer(self) -> BaseTransform:
        return transforms.Scale(factor=self.factor, interpolation=self.interpolation)


class SharpenTransformer(AugLyImageTransformer):
    """
    Alters the sharpness of an image
    """

    def __init__(self, factor: ContinuousValue(-10, 10)):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Sharpen(factor=self.factor)


class ShufflePixelsTransformer(AugLyImageTransformer):
    """
    Shuffles the pixels of an image with respect to the shuffling factor. The
    factor denotes percentage of pixels to be shuffled and randomly selected
    Note: The actual number of pixels will be less than the percentage given
    due to the probability of pixels staying in place in the course of shuffling
    """

    def __init__(self, factor: ContinuousValue(-10, 10)):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.ShufflePixels(factor=self.factor)
