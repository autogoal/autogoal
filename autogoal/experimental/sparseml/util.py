from autogoal.kb import SemanticType
from tensorflow.keras.models import Model

class KerasModel(SemanticType):
    """Semantic type for a Keras Model"""

    @classmethod
    def _match(cls, x):
        try:
            return isintance(x, Model)
        except:
            return False


