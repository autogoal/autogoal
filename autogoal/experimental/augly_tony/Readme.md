## What is AugLy

[AugLy](https://github.com/facebookresearch/AugLy) is a data augmentations library that currently supports four modalities (audio, image, text and video) and over 100 augmentations. Due to the usefulness of increasing the data when training a model or evaluating its robustness, moust transformers for **text**, **images** and **audio** were included using the resources provided by this framework.

## Submodule ``transformers``

The transformers module contains the integration of the framework to autogoal, which has the following distribution:

- ``_text.py`` : Contains the definitions for the text augments
- ``_images.py`` : Contains the definitions for the image augments
- ``_audio.py`` : Contains the definitions for the audio augments

## Submodule `semantic`

Contains the `Image` and `Audio` semantic types.

The semantic type `Image` is based on `PIL.Image.Image` and `Audio` is based on `numpy.ndarray`.



## Install dependency

```bash
sudo apt-get install python3-magic
pip install -U augly 
```

Also with ``example.py`` you have an example of how to use the transformers in the pipeline. 