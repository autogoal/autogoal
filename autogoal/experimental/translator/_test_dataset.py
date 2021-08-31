# Gelin Eguinosa Rosique

import pickle
from os.path import join


# Location of the Documents and their translations.
docs_folder = '/home/coder/autogoal/autogoal/experimental/translator/test_documents'
trans_folder = '/home/coder/autogoal/autogoal/experimental/translator/test_translations'
docs_index_file = 'index_documents.pickle'
trans_index_file = 'index_translations.pickle'

def load():
    """
    Load 50 examples of translations using the Translator from English,
    German and French to Spanish.

    The languages of the Documents and their corresponding translation:
    01-10 -> Deutsch to Spanish
    11-20 -> English to Spanish
    21-30 -> French to Spanish
    31-35 -> English & German to Spanish
    36-40 -> English & French to Spanish
    41-45 -> German & French to Spanish
    46-50 -> English, German & French to Spanish
    """
    # Load the documents' index:
    docs_index_path = join(docs_folder, docs_index_file)
    with open(docs_index_path, 'rb') as file:
        docs_index = pickle.load(file)

    # Load the translations' index:
    trans_index_path = join(trans_folder, trans_index_file)
    with open(trans_index_path, 'rb') as file:
        trans_index = pickle.load(file)

    # Create a list with the text of the Documents & Translations
    documents = []
    translations = []
    for doc_id in docs_index:
        # Load the content of the document
        doc_file_name = docs_index[doc_id]
        document_path = join(docs_folder, doc_file_name)
        with open(document_path, 'r') as file:
            doc_text = file.read()

        # Load the content of the translation
        trans_file_name = trans_index[doc_id]
        translation_path = join(trans_folder, trans_file_name)
        with open(translation_path, 'r') as file:
            trans_text = file.read()

        # Add the contents to the lists
        documents.append(doc_text)
        translations.append(trans_text)

    # Return the Documents and Translations as Training and Test
    X_train = documents[:]
    y_train = translations[:]
    X_test = documents
    y_test = translations

    return X_train, y_train, X_test, y_test
