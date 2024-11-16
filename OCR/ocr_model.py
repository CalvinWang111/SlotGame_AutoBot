"""
Script for class COR
"""
from paddleocr import PaddleOCR

class OCRmodel(PaddleOCR):
    """
    class of OCR
    """
    def __init__(self, use_angle_cls=True, lang='ch'):
        super().__init__(use_angle_cls=use_angle_cls, lang=lang)
    
    def is_freegame(self, image_path) -> bool:
        """
        Determine whether it enters the free game or the base game
        """

    def test(self, image_path):
        return self.ocr(image_path)