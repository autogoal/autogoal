# Gelin Eguinosa Rosique

from autogoal.kb import AlgorithmBase, Document, Text
from _multi_translator import MultiTranslator


class Translator(AlgorithmBase):
    """
    Class to translate documents to Spanish. It receives a Document, detects the
    languages present at the documents and translates its content to Spanish.
    """

    def __init__(self):
        """
        Creates an instance of the class MultiTranslator with Spanish as the
        target language.
        """
        self.lang_model = MultiTranslator('es')

    def run(self, text: Text) -> Text:
        """
        Pass the document to the instance of MultiTranslator to process its
        content, detect the languages present at the document at translate its
        content to Spanish.

        Params:
        text -> The text were are going to translate to Spanish.
        """
        translation = self.lang_model.translation(text)
        return translation
