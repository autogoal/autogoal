# Auto GOAL permite a los investigadores y profesionales desarrollar
# rápidamente algoritmos de referencia optimizados en diversos
# problemas de aprendizaje automático.
# Para utilizar AutoGOAL de la forma más sencilla posible es necesario definir 3 componentes:
# - Entrada
# - Salida
# - Métrica a optimizar (Función objetivo)
# Que se le definen a la clase AutoML

# El primer paso es importar la clase AutoML

from autogoal.ml import AutoML

# El segundo paso es tener un dataset representado en alguno de los tipos definidos en AutoGOAL.
# Por ejemplo en este caso utilizaremos HAHA un corpus de mensajes de Twitter en español
# que queremos clasificar en humorísticos o no.

from autogoal.datasets import haha

# Cargando los datos
X_train, y_train, X_test, y_test = haha.load()

# Como tenemos que representarlo según los tipos definidos en AutoGOAL
# Vamos a importar los tipos que nos hacen falta para HAHA
# En este caso podemos verlo como un listado (que se representa con el tipo Seq de Sequence)
# de oraciones (que se representa con el tipo Sentence) , ya que los mensajes son muy cortos.
# De las que conocemos (al menos para una parte para modelar como supervisado con el tipo Supervised)
# la categoría de humor (esta categoría podemos representarla con el tipo VectorCategorical).

# Definir el tipo de entrada como las oraciones + las clases supervisadas de las mismas, le deja claro a
# AutoGOAL que queremos resolver el problema de forma supervisada con un entrenamiento.

from autogoal.kb import Seq, Sentence, VectorCategorical, Supervised

# ¿Cómo utilizamos esto en la clase AutoML?
automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output=VectorCategorical,  # **tipo de salida**
    # tenemos el parámetro score_metric  para definir la función objetivo,
    # que si no le fijamos un valor utiliza por defecto la función `autogoal.ml.metrics.accuracy`.
)

# Ya hasta aquí hemos definido el problema que queremos resolver
# ahora solo nos resta ejecutar nuestro algoritmo, llamando al método `fit`.

# Para monitorear el estado del proceso de AutoML, podemos pasar un logger al método `fit`.
from autogoal.search import RichLogger

# Entrenando...
automl.fit(X_train, y_train, logger=RichLogger())

# Conociemdo que tan bueno es nuestro algoritmo
score = automl.score(X_test, y_test)
print(f"Score: {score:0.3f}")

# Esto significa que nuestro algoritmo el mejor pipeline que encontró reportó un accuracy "result"

# También puede llamarse al método predict que nos hace la predicción para un conjunto de ejemplos

# Prediciendo...
predictions = automl.predict(X_test)

for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
    print(sentence, "-->", real, "vs", predicted)
