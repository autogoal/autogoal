# Gelin Eguinosa Rosique

import fasttext
import requests
from os import mkdir
from os.path import isdir, isfile, join


class Detector:
    """
    Class to find the spoken language in a text using a Fasttext Language Model.
    """
    # Data Locations
    data_folder = '/home/coder/autogoal/autogoal/experimental/translator/data'
    fasttext_model_file = 'fasttext_model[lid.176].bin'
    fasttext_model_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'

    def __init__(self):
        """
        Check if the Fasttext Language Model is accessible in a local file,
        download it if not. Load the Language Model.
        """
        # Fasttext Model Path
        fasttext_model_path = join(self.data_folder, self.fasttext_model_file)

        # Check if the data folder exists, if not, create it.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)

        # Check if the Fasttext Language Model exists, if not, download it.
        if not isfile(fasttext_model_path):
            # Download Model & save it.
            request = requests.get(self.fasttext_model_url, allow_redirects=True)
            with open(fasttext_model_path, 'wb') as file:
                file.write(request.content)

        # Load the Fasttext Language Model
        self.lang_model = fasttext.load_model(fasttext_model_path)

    def get_language(self, text):
        """
        Determine the language of a text and return the abbreviated name of the
        language (a.k.a. 'es' for Spanish, 'en' for English). Return list of the
        language codes if it receives a list of texts.
        :param text: A string, or a list of strings, representing the text.
        :return: A string, or a list of strings, with the abbreviated names of
        the languages found in the text.
        """
        # Check if we are receiving a list
        if type(text) == list:
            return self._get_languages(text)

        # We received only one text
        lang_tag = self._get_languages([text])[0]
        return lang_tag

    def _get_languages(self, texts):
        """
        Given a list of texts, determine the languages spoken in them. The
        result is as a list with the two character representation of the
        language ('es' for Spanish, 'en' for English, etc..).
        This function assumes that each text represents a single line of text.
        :param texts: Sequence of texts.
        :return: A list with representation of the languages spoken in each of
        the texts.
        """
        # List where the languages found will be saved
        lang_tags = []

        # Iterate through the texts and get their languages
        for text in texts:
            # Delete all the newlines to avoids error in using fasttext
            text.replace('\n', ' ')

            # Predict the Language for the text
            languages, probabilities = self.lang_model.predict(text)
            language = languages[0].split('__')[-1]
            lang_tags.append(language)

        # Return the languages found
        return lang_tags
