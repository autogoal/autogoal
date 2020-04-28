import spacy

from autogoal.grammar import Categorical, Boolean
from autogoal.kb import Sentence, Tuple, Word, Flags, List
from autogoal.utils import nice_repr


@nice_repr
class SpacyNLP:
    def __init__(
        self,
        language: Categorical("en", "es"),
        extract_pos: Boolean(),
        extract_lemma: Boolean(),
        extract_pos_tag: Boolean(),
        extract_dep: Boolean(),
        extract_entity: Boolean(),
        extract_details: Boolean(),
        extract_sentiment: Boolean(),
    ):
        self.language = language
        self.extract_pos = extract_pos
        self.extract_lemma = extract_lemma
        self.extract_pos_tag = extract_pos_tag
        self.extract_dep = extract_dep
        self.extract_entity = extract_entity
        self.extract_details = extract_details
        self.extract_sentiment = extract_sentiment
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load(self.language)

        return self._nlp

    def run(self, input: Sentence()) -> Tuple(List(Word()), List(Flags())):
        tokenized = self.nlp(input)

        tokens = []
        flags = []

        for token in tokenized:
            token_flags = {}
            if self.extract_lemma:
                token_flags["lemma"] = token.lemma_
            if self.extract_pos_tag:
                token_flags["pos"] = token.pos_

                for kv in token.tag_.split("|"):
                    kv = kv.split("=")
                    if len(kv) == 2:
                        token_flags["tag_" + kv[0]] = kv[1]
                    else:
                        token_flags["tag_" + kv[0]] = True

            if self.extract_dep:
                token_flags["dep"] = token.dep_
            if self.extract_entity:
                token_flags["ent_type"] = token.ent_type_
                token_flags["ent_kb_id"] = token.ent_kb_id_
            if self.extract_details:
                token_flags["is_alpha"] = token.is_alpha
                token_flags["is_ascii"] = token.is_ascii
                token_flags["is_digit"] = token.is_digit
                token_flags["is_lower"] = token.is_lower
                token_flags["is_upper"] = token.is_upper
                token_flags["is_title"] = token.is_title
                token_flags["is_punct"] = token.is_punct
                token_flags["is_left_punct"] = token.is_left_punct
                token_flags["is_right_punct"] = token.is_right_punct
                token_flags["is_space"] = token.is_space
                token_flags["is_bracket"] = token.is_bracket
                token_flags["is_quote"] = token.is_quote
                token_flags["is_currency"] = token.is_currency
                token_flags["like_url"] = token.like_url
                token_flags["like_num"] = token.like_num
                token_flags["like_email"] = token.like_email
                token_flags["is_oov"] = token.is_oov
                token_flags["is_stop"] = token.is_stop
            if self.extract_sentiment:
                token_flags["sentiment"] = token.sentiment

            tokens.append(token.text)
            flags.append(token_flags)

        return tokens, flags
