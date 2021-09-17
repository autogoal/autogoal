from autogoal.kb._semantics import SemanticType
from PIL.Image import Image as img
from numpy import ndarray
from os import path
import sndhdr



class Audio(SemanticType):
    '''
    Define a semantic type for audios or sounds, currently support:
    
    • path to the sound file as a `str`

    • `ndarray` from `numpy`
    '''


    @classmethod
    def _match(cls, x) -> bool:
        try:
            if isinstance(x,str):
                if not path.isfile(x):
                    return False
                return sndhdr.what(path.split[1]) is not None   # confirm that this file is actually an audio
        except TypeError:
            return False

        return  isinstance(x, ndarray) or super()._match(x)
       
   
