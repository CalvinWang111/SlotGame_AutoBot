import cv2
from pathlib import Path
import numpy as np

template_dir = Path('./images/golden/symbols/base_game')

for path in template_dir.glob('*.png'):
    template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    # template should be BGRA if it has an alpha channel
    if template.shape[2] < 4:
        print(f"{path.stem} does not have an alpha channel.")
        continue
    
    alpha_channel = template[:, :, 3]
    height, width = alpha_channel.shape

    # Calculate how many pixels are “nearly transparent” (e.g. alpha < 32)
    total_pixels = alpha_channel.size
    transparent_pixels = np.sum(alpha_channel < 32)
    transparent_ratio = transparent_pixels / total_pixels
    print(f'{path.stem} - Transparent ratio: {transparent_ratio:.2f}')

    # ---- Create a new image to visualize the alpha channel ----
    # Option A: Grayscale visualization (3-channel)
    #           Purely replicating the alpha channel into B, G, and R.
    #           Black = alpha=0, White = alpha=255.
    # alpha_3ch = cv2.merge([alpha_channel, alpha_channel, alpha_channel])

    # Optionally, you can normalize alpha to 0–255 for better contrast, 
    # especially if alpha values do not span the full 0–255 range:
    # alpha_normalized = cv2.normalize(alpha_channel, None, 0, 255, cv2.NORM_MINMAX)
    # alpha_3ch = cv2.merge([alpha_normalized, alpha_normalized, alpha_normalized])

    # Option B: Keep it BGRA but let the color channels be the alpha for visualization.
    new_image = np.dstack((alpha_channel, alpha_channel, alpha_channel, alpha_channel))

    # Here we'll just show a 3-channel grayscale image of the alpha:
    cv2.imshow('Original Template', template)
    cv2.imshow('Alpha Visualization', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()