from autogoal.kb import Sentence, Seq, Supervised ,VectorCategorical
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from autogoal.utils import Min, Gb

from autogoal.experimental.fasttex.datasets.text_classification import load
X_train, y_train , X_test , y_test = load()
            

from autogoal.experimental.fasttex._base import  SupervisedTextClassifier
automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output= VectorCategorical,  # **tipo de salida**    
    registry= [SupervisedTextClassifier]+find_classes() ,
    evaluation_timeout= 30 * Min,
    memory_limit=3.5 * Gb,
    search_timeout= 2 * Min,
    #errors="raise"
)

from autogoal.search import RichLogger
automl.fit(X_train,y_train,logger=RichLogger())
score = automl.score(X_test, y_test)
print(score)




from autogoal.experimental.fasttex._base import UnsupervisedWordRepresentationPT

# Descargando los modelos preentrenados en idioma espanhol
UnsupervisedWordRepresentationPT.download(lang='es') 

# Usando el modelo con fuente en wikipedia en espanhol, y vectores de tamanho 250
uwr = UnsupervisedWordRepresentationPT(250, 'wiki', 'es') 

# Transformando palabras a vectores
vectors = uwr.run(["hola", "mundo", "planeta"])

print(vectors)


# Luego de hacer uwr.fit() o uwr.run() la instancia contiene el modelo en el atributo model
print(uwr.model.words)

# Y usar metodos de esa instancia tales como like get_nearest_neighbors() y get_analogies()
print(uwr.model.get_nearest_neighbors("vegetal"))
print(uwr.model.get_analogies("cuchara", "tenedor", "escudo"))
