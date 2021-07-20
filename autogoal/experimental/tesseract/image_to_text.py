from autogoal.experimental.tesseract.semantic import Image_File
from autogoal.kb import AlgorithmBase
from autogoal.kb._semantics import Text, Seq
from PIL import Image
import pytesseract

class TesseractImageToText(AlgorithmBase):
    def __init__(
        self,
        lang=None,
        config='',
        nice=0,
        timeout=0,
    ):
        self.lang=lang
        self.config=config
        self.nice=nice
        self.timeout=timeout
        super().__init__()
    
    
    def run(self, image_dir: Image_File) -> Text:
        image = Image.open(image_dir)
            
        text = pytesseract.image_to_string(
            image,
            lang = self.lang, 
            config = self.config,
            nice = self.nice, 
            timeout = self.timeout
        )
        return text