import numpy as np
from pathlib import Path
import cv2

class Template:
    def __init__(self, path:Path, img:np.ndarray):
        self.path = path
        self.name = path.stem
        self.img = img
        self.best_scale = None
        self.is_square = False