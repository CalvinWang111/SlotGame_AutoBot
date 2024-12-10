import cv2
from pathlib import Path
import numpy as np

template_dir = Path('./images/fu/symbols/base_game')

for path in template_dir.glob('*.png'):
    template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    alpha_channel = template[:, :, 3]
    total_pixels = alpha_channel.size
    transparent_pixels = np.sum(alpha_channel < 32)
    transparent_ratio = transparent_pixels / total_pixels
    # if transparent_ratio < 0.1:
    #     square_tempaltes.append({"name":path.stem, "template":template})
    # else:
    #     non_square_templates.append({"name":path.stem, "template":template})
    print(f'{path.stem}: {transparent_ratio}')
    cv2.imshow('template', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cell height: 200
# 176
# 141
# 145
# 188
# 136
# 193

# range: 0.85 ~ 1.15
# step: 0.03
