# Gelin Eguinosa Rosique

from os.path import isdir, join
from sentence_transformers import SentenceTransformer, util


# Data locations
data_folder = '/home/coder/autogoal/autogoal/experimental/translator/data'
sentence_model_folder = 'sentence_transformers_model'


def text_similarity(text1, text2):
    """
    Compute the Semantic Similarity of two text, comparing the similarity of
    their sentences using the library sentence_transformers.
    :param text1: A string or a list of strings representing the first text.
    :param text2: A string or a list of strings representing the second text.
    :return: The Sum of the cosine similarity between the sentences of the two
    texts.
    """
    # Load the SentenceTransformer Model.
    # Check if it's available locally, otherwise download it.
    sentence_model_path = join(data_folder, sentence_model_folder)
    if isdir(sentence_model_path):
        model = SentenceTransformer(sentence_model_path)
    else:
        # Download the model
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Get the sentences of each of the texts.
    sentences_text1 = _text_into_sentences(text1)
    sentences_text2 = _text_into_sentences(text2)

    # Get the embeddings for all the sentences.
    embeddings_text1 = model.encode(sentences_text1)
    embeddings_text2 = model.encode(sentences_text2)

    # Store the sum of all the cosine similarities:
    total_cos_sim = 0
    # The maximum number of sentences we compare:
    min_length = min(len(sentences_text1), len(sentences_text2))
    # Get the similarities between sentences and add them to the total.
    for i in range(min_length):
        # Cosine similarity between the i sentences
        sent1 = sentences_text1[i]
        sent2 = sentences_text2[i]
        cos_sim = util.cos_sim(embeddings_text1[i], embeddings_text2[i])
        total_cos_sim += float(cos_sim[0][0])

    # The similarity between both text would the average of all the similarities
    # calculated.
    similarity = total_cos_sim/min_length
    return similarity


def _text_into_sentences(text):
    """
    Transforms a text into a list of sentences.
    :param text: A string or list of strings representing the text.
    :return: A list with the sentences inside the text.
    """
    # Where we are going to store the sentences of the text
    text_sentences = []

    # Check if we are dealing with a list
    if type(text) == list:
        # Go through each of the elements in the list and call this method
        # recursively
        for line in text:
            # Check that the line is a string.
            if not type(line) == str:
                raise Exception("We only accept strings or lists of strings.")
            # Transform the line into sentences
            sentences = _text_into_sentences(line)
            # Add them the list of sentences of the text
            text_sentences += sentences
        # Return the union of all the sentences in the text
        return text_sentences

    # If we are working with a string. First split it by new lines.
    lines = text.split('\n')
    for line in lines:
        # Split each line by sentences.
        sentences = line.split('.')
        # Add each one to the sentences' list.
        for sentence in sentences:
            # Check the sentences are not empty
            sentence = sentence.strip()
            if sentence:
                text_sentences.append(sentence)
    # Return the list with all the sentences.
    return text_sentences
