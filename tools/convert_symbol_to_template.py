import numpy as np
import cv2
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

def convert_symbol_to_template(symbol_path, template_path):
    template = cv2.imread(str(symbol_path), cv2.IMREAD_UNCHANGED)
    
    if template.shape[2] == 3:
        alpha_channel = np.full((template.shape[0], template.shape[1]), 255, dtype=template.dtype)
        template = np.dstack((template, alpha_channel))
    
    # Create a blank mask and draw the largest contour
    alpha_channel = template[:, :, 3]
    _, thresh = cv2.threshold(alpha_channel, 32, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    eroded_mask = cv2.erode(mask, kernel, iterations=3)
    x, y, w, h = cv2.boundingRect(eroded_mask)
    template = template[y:y+h, x:x+w]
    
    cv2.imwrite(str(template_path), template)

base_symbol_dir = root_dir / 'images' / 'fu' / 'symbols' / 'base_game'
free_symbol_dir = root_dir / 'images' / 'fu' / 'symbols' / 'free_game'

for base_symbol_path in base_symbol_dir.glob('*.png'):
    base_template_path = base_symbol_dir / base_symbol_path.name
    convert_symbol_to_template(base_symbol_path, base_template_path)
    
for free_symbol_path in free_symbol_dir.glob('*.png'):
    free_template_path = free_symbol_dir / free_symbol_path.name
    convert_symbol_to_template(free_symbol_path, free_template_path)