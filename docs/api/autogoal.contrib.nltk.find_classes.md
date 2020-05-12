# `autogoal.contrib.nltk.find_classes`

> [📝](/usr/lib/python3/dist-packages/autogoal/contrib/nltk/__init__.py#L5)
> `find_classes(include='.*', exclude=None)`

Returns the list of all `nltk` wrappers in `autogoal`.

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
>>> pprint(find_classes(include='.*Stemmer', exclude='.*Lancaster.*'))
[<class 'autogoal.contrib.nltk._generated.ISRIStemmer'>,
 <class 'autogoal.contrib.nltk._generated.PorterStemmer'>,
 <class 'autogoal.contrib.nltk._generated.RSLPStemmer'>,
 <class 'autogoal.contrib.nltk._generated.SnowballStemmer'>]

```
