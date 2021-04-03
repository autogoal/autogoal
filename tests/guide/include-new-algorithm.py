# ¿Qué podemos hacer cuando queremos incluir un nuevo algoritmo a AutoGOAL
# y que este se integre de forma natural con todos los otros para poder formar parte de posibles pipelines.

# Primero tenemos que importar la clase de la que debemos heredar para que se reaproveche todo el sistema de
# conectar pipelines de AutoGOAL

from autogoal.kb import AlgorithmBase

from autogoal.grammar import BooleanValue, DiscreteValue
from autogoal.kb import *

#  Tenemos que crear una clase que representa a nuestro nuevo algoritmo

# Supongamos que queremos un Algoritmo que puede conviertir o no a minúsculas y eliminar las palabras muy cortas.
# Al definirlas de esta forma le estamos pidiendo a AutoGOAL que pruebe a hacerlas o no en diferentes pipelines,
# teniendo e cuenta además distintos valores.


class NewAlgorithm(AlgorithmBase):

    # En el constructor tenemos que poner los hiperparámtros que son optimizables
    def __init__(
        self,
        # min_length tomará valores entre cero y cinco de forma automática para diferentes pipelines.
        # Este parámetro está permitiendo buscar distintos tamaños de palabra y probar cual de ellos será mejor
        min_length: DiscreteValue(min=0, max=5),
        # lower es un parámetro que en algunos casos será True y en otros False. Podemos utilizarlo
        # para llevar o no a minúsculas el texto.
        lower: BooleanValue(),
    ):
        self.min_length = min_length
        self.lower = lower

    # El método run es el método que tienen en común todo los algoritmos incluidos en AutoGOAL es el
    # que permite a la biblioteca ejecutar el método.
    # Este método define tanto la entrada como la salida de tu algoritmo,
    # utilizando anotaciones de los tipo semánticos de AutoGOAL.
    # Además es quien contiene el funcionamiento real de nuestro algoritmo, es el código que se ejecutará.

    # Digamos que recibe una oración (Sentence) y devuelve una lista de palabras (Seq[Word])
    def run(self, input: Sentence) -> Seq[Word]:

        # Como podemos ver la implementación del método se realiza bastante independiente de la biblioteca

        if self.lower:
            input = input.lower()

        result = []

        for i in input.split():
            if len(i) > self.min_length:
                result.append(i)

        return result


# Una vez que tenemos listo nuestro algoritmo solo nos queda indicarle a la clase AutoML que lo utilice en la búsqueda

# Estos son algunos import que nos hacen falta más adelante
from autogoal.ml import AutoML
from autogoal.contrib import find_classes

# Probemos con HAHA
from autogoal.datasets import haha

# Cargando los datos
X_train, y_train, X_test, y_test = haha.load()


# Creando la instancia de AutoML con nuestra clase
automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output=VectorCategorical,  # **tipo de salida**
    # Agregando nuestra clase y todo el resto de algortimos de AutoGOAL
    registry=[NewAlgorithm] + find_classes(),
)

# Ahora sencillamente tenemos que ejecutar AutoML y ya nuestro algoritmo aparecerá en algunos pipelines.
# Debemos tener en cuenta que esto no garantiza qeu aparezca en el mejor pipeline encontrado, sino que se conectará
# con el resto de los algoritmo como si fuera nativo de AutoGOAL.

automl.fit(X_train, y_train)

score = automl.score(X_test, y_test)
print(score)
