from autogoal.kb import Matrix, MatrixContinuous, MatrixContinuousDense, MatrixContinuousSparse, build_pipeline_graph, Text, Document, Word, List, Tuple, Sentence
from autogoal.kb import algorithm
from autogoal.ml import AutoML
import pprint

class ExactAlgorithm:
    def run(self, input:MatrixContinuousDense()) -> MatrixContinuousDense():
        pass

class HigherInputAlgorithm:
    def run(self, input:MatrixContinuous()) -> MatrixContinuousDense():
        pass

class LowerOutputAlgorithm:
    def run(self, input:MatrixContinuousDense()) -> MatrixContinuousDense():
        pass
    
class WordToWordAlgorithm:
    def run(self, input:Word()) -> Word():
        pass
    
class TextToWordAlgorithm:
    def run(self, input:Text()) -> Word():
        pass

class WordToWordListAlgorithm:
    def run(self, input:Word()) -> List(Word()):
        pass

class WordListToSentenceAlgorithm:
    def run(self, input:List(Word())) -> Sentence():
        pass

class SentenceListToDocumentAlgorithm:
    def run(self, input:List(Sentence())) -> Document():
        pass

class TextListToDocumentAlgorithm:
    def run(self, input:List(Text())) -> Document():
        pass

def assert_graph(graph, start_out, end_in, nodes_amount):
    """ 
    Assert amount of nodes, adjacents of PipelineStart node
    and in-edges of PipelineEnd node"""
    start_node = list(graph)[1] #PipelineStart node have fixed position
    end_node = [node for node in list(graph) if node.__class__.__name__ == "PipelineEnd"][0] #PipelineEnd node
    
    assert(graph.out_degree(start_node) == start_out)
    assert(graph.in_degree(end_node) == end_in)
    assert(graph.number_of_nodes() == nodes_amount)
    
def assert_pipeline_graph_failed(input, output, registry):
    try:
        pipeline_builder = build_pipeline_graph(input=input
                                                ,output=output
                                                ,registry=registry)
        raise AssertionError("Graph built successfully (expected TypeError)")
    except TypeError as error:
        assert(error.args[0].startswith("No pipelines"))
        
def test_build_pipeline_graph():
    test_meta_pipeline_graph()
    test_simple_pipeline_graph()
    
    #test failed graph generation
    assert_pipeline_graph_failed(Text(), Word(), [])
    assert_pipeline_graph_failed(Text(), 
                                 Document(), 
                                 [WordToWordAlgorithm, 
                                  TextToWordAlgorithm, 
                                  WordToWordListAlgorithm,
                                  SentenceListToDocumentAlgorithm,
                                  TextListToDocumentAlgorithm])

def test_meta_pipeline_graph():
    # Test List algorithm generation
    build_pipeline_graph(input=List(Word()),
                         output=List(Word()),
                         registry=[WordToWordAlgorithm])
    
    # Test Tuple breakdown feature
    build_pipeline_graph(input=Tuple(Word(), Matrix()),
                         output=Text(),
                         registry=[WordToWordAlgorithm])
    
    # Test Tuple breakdown feature and List algorithm generation
    build_pipeline_graph(input=Tuple(List(Word()), Matrix()),
                         output=List(Word()),
                         registry=[WordToWordAlgorithm])

def test_simple_pipeline_graph():
    graph = build_pipeline_graph(input=MatrixContinuousDense()
                                 ,output= MatrixContinuousDense()
                                 ,registry=[ExactAlgorithm, HigherInputAlgorithm, LowerOutputAlgorithm]).graph
    assert_graph(graph, 3, 3, 6)
    
    graph = build_pipeline_graph(input=List(Text())
                                 ,output= Document()
                                 ,registry=[WordToWordAlgorithm, 
                                            TextToWordAlgorithm, 
                                            WordToWordListAlgorithm,
                                            WordListToSentenceAlgorithm,
                                            WordListToSentenceAlgorithm,
                                            SentenceListToDocumentAlgorithm,
                                            TextListToDocumentAlgorithm]).graph
    assert_graph(graph, 2, 2, 12)
    
    graph = build_pipeline_graph(input=List(Word())
                                 ,output=Document()
                                 ,registry=[WordToWordAlgorithm, 
                                            TextToWordAlgorithm, 
                                            WordToWordListAlgorithm,
                                            WordListToSentenceAlgorithm,
                                            WordListToSentenceAlgorithm,
                                            SentenceListToDocumentAlgorithm,
                                            TextListToDocumentAlgorithm]).graph
    assert_graph(graph, 2, 1, 10)