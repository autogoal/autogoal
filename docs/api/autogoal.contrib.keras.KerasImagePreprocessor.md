# `autogoal.contrib.keras.KerasImagePreprocessor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/keras/_base.py#L213)
> `KerasImagePreprocessor(self, featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, vertical_flip)`

Augment a dataset of images by making changes to the original training set.

Applies standard dataset augmentation strategies, such as rotating,
scaling and fliping the image.
Uses the `ImageDataGenerator` class from keras.

The parameter `grow_size` determines how many new images will be created for each original image.
The remaining parameters are passed to `ImageDataGenerator`.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `apply_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L835)
> `apply_transform(self, x, transform_parameters)`

Applies a transformation to an image according to given parameters.

# Arguments
    x: 3D tensor, single image.
    transform_parameters: Dictionary with string - parameter pairs
        describing the transformation.
        Currently, the following parameters
        from the dictionary are used:
        - `'theta'`: Float. Rotation angle in degrees.
        - `'tx'`: Float. Shift in the x direction.
        - `'ty'`: Float. Shift in the y direction.
        - `'shear'`: Float. Shear angle in degrees.
        - `'zx'`: Float. Zoom in the x direction.
        - `'zy'`: Float. Zoom in the y direction.
        - `'flip_horizontal'`: Boolean. Horizontal flip.
        - `'flip_vertical'`: Boolean. Vertical flip.
        - `'channel_shift_intensity'`: Float. Channel shift intensity.
        - `'brightness'`: Float. Brightness shift intensity.

# Returns
    A transformed version of the input (same shape).
### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L905)
> `fit(self, x, augment=False, rounds=1, seed=None)`

Fits the data generator to some sample data.

This computes the internal data stats related to the
data-dependent transformations, based on an array of sample data.

Only required if `featurewise_center` or
`featurewise_std_normalization` or `zca_whitening` are set to True.

When `rescale` is set to a value, rescaling is applied to
sample data before computing the internal data stats.

# Arguments
    x: Sample data. Should have rank 4.
     In case of grayscale data,
     the channels axis should have value 1, in case
     of RGB data, it should have value 3, and in case
     of RGBA data, it should have value 4.
    augment: Boolean (default: False).
        Whether to fit on randomly augmented samples.
    rounds: Int (default: 1).
        If using data augmentation (`augment=True`),
        this is how many augmentation passes over the data to use.
    seed: Int (default: None). Random seed.
### `flow`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L368)
> `flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)`

Takes data & label arrays, generates batches of augmented data.

# Arguments
    x: Input data. NumPy array of rank 4 or a tuple.
        If tuple, the first element
        should contain the images and the second element
        another NumPy array or a list of NumPy arrays
        that gets passed to the output
        without any modifications.
        Can be used to feed the model miscellaneous data
        along with the images.
        In case of grayscale data, the channels axis of the image array
        should have value 1, in case
        of RGB data, it should have value 3, and in case
        of RGBA data, it should have value 4.
    y: Labels.
    batch_size: Int (default: 32).
    shuffle: Boolean (default: True).
    sample_weight: Sample weights.
    seed: Int (default: None).
    save_to_dir: None or str (default: None).
        This allows you to optionally specify a directory
        to which to save the augmented pictures being generated
        (useful for visualizing what you are doing).
    save_prefix: Str (default: `''`).
        Prefix to use for filenames of saved pictures
        (only relevant if `save_to_dir` is set).
    save_format: one of "png", "jpeg"
        (only relevant if `save_to_dir` is set). Default: "png".
    subset: Subset of data (`"training"` or `"validation"`) if
        `validation_split` is set in `ImageDataGenerator`.

# Returns
    An `Iterator` yielding tuples of `(x, y)`
        where `x` is a NumPy array of image data
        (in the case of a single image input) or a list
        of NumPy arrays (in the case with
        additional inputs) and `y` is a NumPy array
        of corresponding labels. If 'sample_weight' is not None,
        the yielded tuples are of the form `(x, y, sample_weight)`.
        If `y` is None, only the NumPy array `x` is returned.
### `flow_from_dataframe`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L546)
> `flow_from_dataframe(self, dataframe, directory=None, x_col='filename', y_col='class', weight_col=None, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', validate_filenames=True, **kwargs)`

Takes the dataframe and the path to a directory
 and generates batches of augmented/normalized data.

**A simple tutorial can be found **[here](
                            http://bit.ly/keras_flow_from_dataframe).

# Arguments
    dataframe: Pandas dataframe containing the filepaths relative to
        `directory` (or absolute paths if `directory` is None) of the
        images in a string column. It should include other column/s
        depending on the `class_mode`:
        - if `class_mode` is `"categorical"` (default value) it must
            include the `y_col` column with the class/es of each image.
            Values in column can be string/list/tuple if a single class
            or list/tuple if multiple classes.
        - if `class_mode` is `"binary"` or `"sparse"` it must include
            the given `y_col` column with class values as strings.
        - if `class_mode` is `"raw"` or `"multi_output"` it should contain
        the columns specified in `y_col`.
        - if `class_mode` is `"input"` or `None` no extra column is needed.
    directory: string, path to the directory to read images from. If `None`,
        data in `x_col` column should be absolute paths.
    x_col: string, column in `dataframe` that contains the filenames (or
        absolute paths if `directory` is `None`).
    y_col: string or list, column/s in `dataframe` that has the target data.
    weight_col: string, column in `dataframe` that contains the sample
        weights. Default: `None`.
    target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
        The dimensions to which all images found will be resized.
    color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to have 1 or 3 color channels.
    classes: optional list of classes (e.g. `['dogs', 'cats']`).
        Default: None. If not provided, the list of classes will be
        automatically inferred from the `y_col`,
        which will map to the label indices, will be alphanumeric).
        The dictionary containing the mapping from class names to class
        indices can be obtained via the attribute `class_indices`.
    class_mode: one of "binary", "categorical", "input", "multi_output",
        "raw", sparse" or None. Default: "categorical".
        Mode for yielding the targets:
        - `"binary"`: 1D NumPy array of binary labels,
        - `"categorical"`: 2D NumPy array of one-hot encoded labels.
            Supports multi-label output.
        - `"input"`: images identical to input images (mainly used to
            work with autoencoders),
        - `"multi_output"`: list with the values of the different columns,
        - `"raw"`: NumPy array of values in `y_col` column(s),
        - `"sparse"`: 1D NumPy array of integer labels,
        - `None`, no targets are returned (the generator will only yield
            batches of image data, which is useful to use in
            `model.predict_generator()`).
    batch_size: size of the batches of data (default: 32).
    shuffle: whether to shuffle the data (default: True)
    seed: optional random seed for shuffling and transformations.
    save_to_dir: None or str (default: None).
        This allows you to optionally specify a directory
        to which to save the augmented pictures being generated
        (useful for visualizing what you are doing).
    save_prefix: str. Prefix to use for filenames of saved pictures
        (only relevant if `save_to_dir` is set).
    save_format: one of "png", "jpeg"
        (only relevant if `save_to_dir` is set). Default: "png".
    follow_links: whether to follow symlinks inside class subdirectories
        (default: False).
    subset: Subset of data (`"training"` or `"validation"`) if
        `validation_split` is set in `ImageDataGenerator`.
    interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image.
        Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
        If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
        supported. If PIL version 3.4.0 or newer is installed, `"box"` and
        `"hamming"` are also supported. By default, `"nearest"` is used.
    validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this
        option can lead to speed-up in the execution of this function.
        Default: `True`.

# Returns
    A `DataFrameIterator` yielding tuples of `(x, y)`
    where `x` is a NumPy array containing a batch
    of images with shape `(batch_size, *target_size, channels)`
    and `y` is a NumPy array of corresponding labels.
### `flow_from_directory`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L437)
> `flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')`

Takes the path to a directory & generates batches of augmented data.

# Arguments
    directory: string, path to the target directory.
        It should contain one subdirectory per class.
        Any PNG, JPG, BMP, PPM or TIF images
        inside each of the subdirectories directory tree
        will be included in the generator.
        See [this script](
        https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
        for more details.
    target_size: Tuple of integers `(height, width)`,
        default: `(256, 256)`.
        The dimensions to which all images found will be resized.
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    classes: Optional list of class subdirectories
        (e.g. `['dogs', 'cats']`). Default: None.
        If not provided, the list of classes will be automatically
        inferred from the subdirectory names/structure
        under `directory`, where each subdirectory will
        be treated as a different class
        (and the order of the classes, which will map to the label
        indices, will be alphanumeric).
        The dictionary containing the mapping from class names to class
        indices can be obtained via the attribute `class_indices`.
    class_mode: One of "categorical", "binary", "sparse",
        "input", or None. Default: "categorical".
        Determines the type of label arrays that are returned:
        - "categorical" will be 2D one-hot encoded labels,
        - "binary" will be 1D binary labels,
            "sparse" will be 1D integer labels,
        - "input" will be images identical
            to input images (mainly used to work with autoencoders).
        - If None, no labels are returned
          (the generator will only yield batches of image data,
          which is useful to use with `model.predict_generator()`).
          Please note that in case of class_mode None,
          the data still needs to reside in a subdirectory
          of `directory` for it to work correctly.
    batch_size: Size of the batches of data (default: 32).
    shuffle: Whether to shuffle the data (default: True)
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    save_to_dir: None or str (default: None).
        This allows you to optionally specify
        a directory to which to save
        the augmented pictures being generated
        (useful for visualizing what you are doing).
    save_prefix: Str. Prefix to use for filenames of saved pictures
        (only relevant if `save_to_dir` is set).
    save_format: One of "png", "jpeg"
        (only relevant if `save_to_dir` is set). Default: "png".
    follow_links: Whether to follow symlinks inside
        class subdirectories (default: False).
    subset: Subset of data (`"training"` or `"validation"`) if
        `validation_split` is set in `ImageDataGenerator`.
    interpolation: Interpolation method used to
        resample the image if the
        target size is different from that of the loaded image.
        Supported methods are `"nearest"`, `"bilinear"`,
        and `"bicubic"`.
        If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
        supported. If PIL version 3.4.0 or newer is installed,
        `"box"` and `"hamming"` are also supported.
        By default, `"nearest"` is used.

# Returns
    A `DirectoryIterator` yielding tuples of `(x, y)`
        where `x` is a NumPy array containing a batch
        of images with shape `(batch_size, *target_size, channels)`
        and `y` is a NumPy array of corresponding labels.
### `get_random_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L745)
> `get_random_transform(self, img_shape, seed=None)`

Generates random parameters for a transformation.

# Arguments
    seed: Random seed.
    img_shape: Tuple of integers.
        Shape of the image that is transformed.

# Returns
    A dictionary containing randomly chosen parameters describing the
    transformation.
### `random_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L892)
> `random_transform(self, x, seed=None)`

Applies a random transformation to an image.

# Arguments
    x: 3D tensor, single image.
    seed: Random seed.

# Returns
    A randomly transformed version of the input (same shape).
### `standardize`

> [📝](/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py#L690)
> `standardize(self, x)`

Applies the normalization configuration in-place to a batch of inputs.

`x` is changed in-place since the function is mainly used internally
to standardize images and feed them to your network. If a copy of `x`
would be created instead it would have a significant performance cost.
If you want to apply this method without changing the input in-place
you can call the method creating a copy before:

standardize(np.copy(x))

# Arguments
    x: Batch of inputs to be normalized.

# Returns
    The inputs, normalized.
