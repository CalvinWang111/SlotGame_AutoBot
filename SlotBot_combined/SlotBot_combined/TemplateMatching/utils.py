import json
from pathlib import Path
from collections import OrderedDict
import cv2
import pickle
# import torch
import random
import numpy as np

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_object(fname):
    try:
        with open(fname, 'rb') as file:
            content = pickle.load(file)
        return content
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading object: {e}")
        return None

def write_object(content, fname):
    try:
        with open(fname, 'wb') as file:
            pickle.dump(content, file)
    except (IOError, pickle.PickleError) as e:
        print(f"Error saving object: {e}")
        
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# def set_seed(seed):
#     torch.manual_seed(seed)  # Set seed for PyTorch
#     torch.cuda.manual_seed(seed)  # Set seed for CUDA (GPU)
#     torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
#     random.seed(seed)  # Set seed for random module
#     np.random.seed(seed)  # Set seed for numpy
    
#     # Ensure that all operations are deterministic on GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False