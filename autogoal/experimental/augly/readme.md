## What is AugLy

[AugLy](https://github.com/facebookresearch/AugLy) is a data augmentations library that currently supports four modalities (audio, image, text and video) and over 100 augmentations. Due to the usefulness of increasing the data when training a model or evaluating its robustness, various transformers for text and images were included using the resources provided by this framework.

## Module ``augly``

The augly module contains the integration of the framework to autogoal, which has the following distribution:
    
- ``_semantics.py`` : Defines a semantic type for images
- ``_utils.py`` : Contains the wrapper to integrate AugLy types to autogoal
- ``_text.py`` : Contains the definitions for the text augments
- ``_images.py`` : Contains the definitions for the image augments

Also with ``example.py`` you have an example of how to use the transformers in the pipeline. The tests are located in the ``test_augly.py`` file.

## Install dependency

```bash
sudo apt-get install python3-magic
pip install -U augly 
```
 