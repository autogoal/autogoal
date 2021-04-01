# Ya sabemos como se utiliza la clase AutoML de la forma más básica
# pero esta clase cuenta con varios parámetros que nos permiten
# personalizar la ejecución a nuestras condiciones.

from autogoal.ml import AutoML

# Utilizando el corpus de HAHA
from autogoal.datasets import haha

# Cargando los datos
X_train, y_train, X_test, y_test = haha.load()

# Cargando los tipos de datos para representar el dataset
from autogoal.kb import Seq, Sentence, VectorCategorical, Supervised

# Vemos ahora que párametros nuevos podemos definirle a la clase AutoML
automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output=VectorCategorical,  # **tipo de salida**
    # el score_metric define la función objetivo a optimizar y puede ser definida por nosotros en un método propio
    score_metric=balanced_accuracy_score,
    # el parámetro registry nos permite seleccionar un conjunto específico de algoritmo a utilizar en nuestra implementación.
    # Si no se define o se pone None se utilizan todos los algorismos disponibles en AutoGOAL.
    registry=None,
    # search_algorithm permite cambiar el algoritmo de optimization que utiliza AutoGOAL, en estos moemntos también está
    # implementada una búsqueda aleatoria o puedes implementar una nueva clase.
    search_algorithm=PESearch,
    # search_iterations se utiliza para definir la cantidad de iteraciones que queremos que haga nuestro algoritmo de búsqueda
    # osea cantidad de generaciones en la búsqued aevolutiva o en el random
    search_iterations=args.iterations,
    # search_kwargs este parámetro se utiliza para pasar opciones adicionales al algoritmo de búsqueda
    search_kwargs=dict(
        # pop_size es el tamaño de la población
        pop_size=args.popsize,
        # search_timeout es el tiempo máximo total que queremos dedicarle a la búsqueda en segundos
        search_timeout=args.global_timeout,
        # evaluation_timeout es el tiempo máximo para un pipeline, si la ejecución del pipeline se pasa de este texto
        # se detenine y se le asigna fitness cero.
        evaluation_timeout=args.timeout,
        # cantidad máxima de RAM por pipeline. Este número debe ser inferior a la RAM del dispositivo donde se ejecute la experimentación
        # para evitar que el despositivo de bloquee.
        memory_limit=args.memory * 1024 ** 3,
    ),
    # cross_validation_steps cantidad de veces que se evalúa cada pipeline
    cross_validation_steps=3,
    # validation_split por ciento del tamaño del training set que se utiliza para cross validation.
    validation_split=0.3,
    # cross_validation es la métrica que se utiliza para mezclar los score de los cross_validation_steps. También está "mean"
    cross_validation="median",
    # random_state es un número para fijar la semilla random de la búsqueda. Esto nos puede ayudar a que aparezcan pipelines similares a los de
    # otra ejecución.
    random_state=None,
    # errors determina que se hce cuando un pipeline lanza una excepción. "warn" lanza un wargnig, "ïgnore" los ignora y
    # "raise" que lanza la excepción y detiene la ejecución.
    errors="warn",
)

# Entrenando.....
automl.fit(X_train, y_train)

# Conociemdo que tan bueno es nuestro algoritmo
result = automl.score(X_test, y_test)
print(result)
