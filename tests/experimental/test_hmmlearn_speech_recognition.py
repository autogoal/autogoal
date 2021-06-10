from autogoal.kb import build_pipeline_graph
from autogoal.experimental.hmmlearn_speech_recognition._manual import HMMLearnSpeechRecognizer
from autogoal.contrib import find_classes
from autogoal.experimental.hmmlearn_speech_recognition.util import AudioFile
from autogoal.kb import Seq, Word, Supervised

def test_algorithm_in_pipeline_graph():
    pipelines = build_pipeline_graph(
        input_types=(Seq[AudioFile], Supervised[Seq[Word]]),
        output_type=Seq[Word],
        registry=find_classes() + [HMMLearnSpeechRecognizer]
    )

    assert HMMLearnSpeechRecognizer in pipelines.nodes()

def test_algorithm_report_correct_types():
    assert HMMLearnSpeechRecognizer.input_types() == (
        Seq[AudioFile], 
        Supervised[Seq[Word]]
    )

    assert HMMLearnSpeechRecognizer.output_type() == Seq[Word]
