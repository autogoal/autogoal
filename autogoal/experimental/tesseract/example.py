from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.experimental.tesseract.semantic import Image_File
from autogoal.experimental.tesseract.dataset import load
from autogoal.experimental.tesseract.image_to_text import TesseractImageToText 
from autogoal.experimental.tesseract.image_to_boxes import TesseractImageToBoxes
from autogoal.experimental.tesseract.image_to_data import TesseractImageToData
from autogoal.experimental.tesseract.similarity import similarity, similarityQuick, similarityRealQuick
from autogoal.contrib import find_classes
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from os import listdir

def test1():
    automl = AutoML(
        input=(Seq[Image_File]),
        output=Seq[Text],
        registry=[TesseractImageToText,TesseractImageToBoxes,TesseractImageToData] + find_classes(),
        evaluation_timeout=Min,
        memory_limit=4 * Gb,
        search_timeout=Min,
        score_metric=similarityRealQuick
    )

    x_train, y_train, x_test, y_test = load()

    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(f"Score 1: {score}")

def test2():
    automl = AutoML(
        input=(Seq[Image_File]),
        output=Seq[FeatureSet],
        registry=[TesseractImageToText,TesseractImageToBoxes,TesseractImageToData] + find_classes(),
        evaluation_timeout=Min,
        memory_limit=4 * Gb,
        search_timeout=Min,
        score_metric=similarityRealQuick
    )

    x_train, y_train, x_test, y_test = load()

    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(f"Score 2: {score}")

test1()
test2()
