# `autogoal.contrib.sklearn.find_classes`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/contrib/sklearn/__init__.py#L37)
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
