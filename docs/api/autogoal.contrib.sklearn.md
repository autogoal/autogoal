# `autogoal.contrib.sklearn`

This module contains wrappers for several estimators and transformers
from `scikit-learn`.

!!! warning
    Importing this module requires `sklearn` with a version equal or greater
    than `0.22`. You can either install it manually or with `pip install autogoal[sklearn]`.

Most of the classes and functions inside this module deal with the automatic
generation of wrappers and thus are considered private API.

The main public functionality exposed by this module is the function
[find_classes](/api/autogoal.contrib.sklearn/#find_classes), which allows to
enumerate the wrappers implemented in this module applying some filters.

!!! note
    You can manually import any wrapper class directly from `autogoal.contrib.sklearn._generated`
    buy beware that namespace changes wildly from version to version and classes in it
    might disappear or change their signature anytime.

## Functions

### `find_classes`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/contrib/sklearn/__init__.py#L37)
> `find_classes(include='.*', exclude=None)`


Returns the list of all `scikit-learn` wrappers in `autogoal`.

You can pass filters to include or exclude specific classes.
The filters are regular expressions that are matched against
the names of the classes. Only classes that pass the `include` filter
and not the `exclude` filter will be returned.
By default all classes are returned.

##### Parameters

- `include`: regular expression to match for including classes. Defaults to `".*"`, i.e., all classes.
- `exclude`: regular expression to match for excluding classes. Defaults to `None`.

##### Examples

```python
>>> from pprint import pprint
>>> pprint(find_classes(include='.*Classifier', exclude='.*Tree.*'))
[<class 'autogoal.contrib.sklearn._generated.KNeighborsClassifier'>,
 <class 'autogoal.contrib.sklearn._generated.PassiveAggressiveClassifier'>,
 <class 'autogoal.contrib.sklearn._generated.RidgeClassifier'>,
 <class 'autogoal.contrib.sklearn._generated.SGDClassifier'>]

```

