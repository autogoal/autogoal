from autogoal.datasets import ag_news as ag_news
from autogoal.utils import Gb, Hour, Sec
from autogoal.ml import AutoML, evaluation_time, accuracy
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
from autogoal.kb import Seq, Supervised, VectorDiscrete, Sentence, VectorCategorical
from autogoal_contrib import find_classes
from autogoal.search import RichLogger, JsonLogger, ConsoleLogger
from autogoal.search import NSPESearch
from autogoal_telegram import TelegramLogger
from autogoal_transformers import WORD_EMB_Distilbert_Base_Uncased, FullFineTunerEmbedderTransformerClassifier,PartialFineTunerEmbedderTransformerClassifier, LORAFineTunerEmbedderTransformerClassifier, TEXT_GEN_Google_T5_T5_Large, GenerativeClassifier

def test_pipeline():
    X_train, y_train, X_test, y_test = ag_news.load(True)

    model = AutoML(
        input=(Seq[Sentence], Supervised[VectorDiscrete]),
        output=VectorDiscrete,
        registry=[FullFineTunerEmbedderTransformerClassifier,PartialFineTunerEmbedderTransformerClassifier,LORAFineTunerEmbedderTransformerClassifier] + find_classes(include="WORD_EMB"),
        objectives=(macro_f1_plain, evaluation_time),
        observations=[("Accuracy", accuracy)],
        search_algorithm=NSPESearch,
        maximize=(True, False),
        evaluation_timeout=4 * Hour,
        search_timeout=96 * Hour,
        memory_limit=30 * Gb,
        cross_validation_steps=3,
        stratified_cross_validation=True,
    )

    loggers = [
        ConsoleLogger(),
        TelegramLogger(
            token="6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
            channel="570734906",
            name="Titan AG News Finetuning Embedders",
            objectives=["Macro F1", ("Eval Time", "Seconds")],
        ),
        JsonLogger("titan-AG-News-finetuning-embedders.json"),
    ]
    
    model.fit(X_train, y_train, logger=loggers)
    results = model.score(X_test, y_test)

    print(f"F1: {results}")


if __name__ == "__main__":
    from autogoal.utils._process import initialize_cuda_multiprocessing
    initialize_cuda_multiprocessing()
    test_pipeline()

