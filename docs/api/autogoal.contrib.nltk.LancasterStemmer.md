# `autogoal.contrib.nltk.LancasterStemmer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L48)
> `LancasterStemmer(self, strip_prefix_flag)`

Lancaster Stemmer

    >>> from nltk.stem.lancaster import LancasterStemmer
    >>> st = LancasterStemmer()
    >>> st.stem('maximum')     # Remove "-um" when word is intact
    'maxim'
    >>> st.stem('presumably')  # Don't remove "-um" when word is not intact
    'presum'
    >>> st.stem('multiply')    # No action taken if word ends with "-ply"
    'multiply'
    >>> st.stem('provision')   # Replace "-sion" with "-j" to trigger "j" set of rules
    'provid'
    >>> st.stem('owed')        # Word starting with vowel must contain at least 2 letters
    'ow'
    >>> st.stem('ear')         # ditto
    'ear'
    >>> st.stem('saying')      # Words starting with consonant must contain at least 3
    'say'
    >>> st.stem('crying')      #     letters and one of those letters must be a vowel
    'cry'
    >>> st.stem('string')      # ditto
    'string'
    >>> st.stem('meant')       # ditto
    'meant'
    >>> st.stem('cement')      # ditto
    'cem'
    >>> st_pre = LancasterStemmer(strip_prefix_flag=True)
    >>> st_pre.stem('kilometer') # Test Prefix
    'met'
    >>> st_custom = LancasterStemmer(rule_tuple=("ssen4>", "s1t."))
    >>> st_custom.stem("ness") # Change s to t
    'nest'
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `parseRules`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/lancaster.py#L182)
> `parseRules(self, rule_tuple=None)`

Validate the set of rules used in this stemmer.

If this function is called as an individual method, without using stem
method, rule_tuple argument will be compiled into self.rule_dictionary.
If this function is called within stem, self._rule_tuple will be used.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L54)
> `run(self, input)`

### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/lancaster.py#L205)
> `stem(self, word)`

Stem a word using the Lancaster stemmer.
        
