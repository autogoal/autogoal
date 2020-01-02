#

# Este .py contiene las clases que ayudan a construir de forma mas o menos semática redes neuronales
#

from .base import register_abstract_class
from .base import register_concrete_class
from .base import Algorithm, BaseObject, BaseAbstract

from keras.layers import Input
from keras.models import Model


class NeuralNetworkModule:
    """ Esta clase representa lo que es común a un elemento de una red neuronal a nivel de código
    """

    def build(self, model):
        raise NotImplementedError()


@register_abstract_class
class PreprocessorModule(BaseAbstract, NeuralNetworkModule):
    """ Representa los posibles elementos para preprocesar los datos en una red neuronal
    """
    pass


@register_abstract_class
class ReductionModule(BaseAbstract, NeuralNetworkModule):
    """ Representa los posibles elementos para reducir dimensiones en los datos ya preprocesados en una red neuronal
    """
    pass


@register_abstract_class
class AbstractFeaturesModule(BaseAbstract, NeuralNetworkModule):
    """ Representa los posibles elementos para descubrir features de mas alto nivel(osea mas abstractos) en una red neuronal
    """
    pass


@register_abstract_class
class ClassificationModule(BaseAbstract, NeuralNetworkModule):
    """ Representa los posibles elementos para un clasificador o regresor en una red neuronal, esta tendría como salida el resultado
    que se está buscando.
    """
    pass


@register_concrete_class
class NeuralNetwork(BaseObject, Algorithm):
    """Representa a todas las posibles redes neuronales
    """

    def __init__(
        self,
        preprocessor: PreprocessorModule,
        reductor: ReductionModule,
        abstractor: AbstractFeaturesModule,
        classifier: ClassificationModule,
    ):
        """Instancia una red neuronal basada en 4 componentes fundamentales:
        Preprocesamiento, Reductor de dimensiones, un componente para descubrir
        features de mas alto nivel (o sea mas abstractos) y una componente
        que sería un clasificador o regresor.
        """
        self.preprocessor = preprocessor
        self.reductor = reductor
        self.abstractor = abstractor
        self.classifier = classifier

    def __repr__(self):
        return "NeuralNetwork(preprocessor=%r, reductor=%r, abstractor=%r, classifier=%r)" % (
            self.preprocessor,
            self.reductor,
            self.abstractor,
            self.classifier
        )

    def compile(self, input_shape):
        input_x = Input(input_shape)
        output_y = self.build(input_x)

        self.model = Model(inputs=input_x, outputs=output_y)


    def build(self, input_x):
        for module in [
            self.preprocessor,
            self.reductor,
            self.abstractor,
            self.classifier
        ]:
            input_x = module.build(input_x)

        return input_x


@register_abstract_class
class BasicClassificationModule(ClassificationModule):
    """
    Un sofmax o clasificador-regresor simple
    """
    pass


@register_concrete_class
class CompositeClassificationModule(BaseObject, ClassificationModule):
    """Compone clases de capas densas y un sofmax o clasificador-regresor detras
    """

    def __init__(self, middle: ClassificationModule, end: BasicClassificationModule):
        self.middle = middle
        self.end = end



    def build(self, input_x):
        y = self.middle.build(input_x)
        z = self.end.build(y)
        return z


@register_concrete_class
class NullModule(BaseObject, PreprocessorModule):
    """Representa un componente que se comporta como una capa pero devuelve por la salida lo mismo que le entra por la entrada.
    """

    def __repr__(self):
        return "NullModule()"

    def build(self, model):
        return model


def build_nn_grammar():
    import autogoal.ontology._generated._keras as kk

    namespace = dict(globals())
    namespace.update(vars(kk))

    grammar = NeuralNetwork.generate_grammar()
    grammar.namespace = namespace

    return grammar
