import cv2
from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 測試OCR大數值
image_dir = Path(r'C:\Users\david\Downloads\大數')
for img_path in image_dir.glob('*.png'):
    img_path = str(img_path)
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    ocr_result = ocr.ocr(img_path, cls=True)
    ocr_result = ocr_result[0]
    for data in ocr_result:
        print(data)

    image = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    # 顯示結果
    plt.show()
