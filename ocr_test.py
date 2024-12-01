import cv2
from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

data = """<position>[63, 55, 98, 13]</position> = <meaning>玩家剩餘金額</meaning>
<position>[195, 129, 83, 18]</position> = <meaning>巨獎金額</meaning>
<position>[101, 175, 47, 15]</position> = <meaning>大獎金額</meaning>
<position>[215, 175, 44, 14]</position> = <meaning>中獎金額</meaning>
<position>[330, 175, 37, 14]</position> = <meaning>小獎金額</meaning>
<position>[84, 838, 25, 12]</position> = <meaning>總押注金額</meaning>
<position>[223, 832, 28, 15]</position> = <meaning>玩家贏得的分數</meaning>"""
examples = re.findall(r"<position>(.*?)</position>.*?<meaning>(.*?)</meaning>", data, re.DOTALL)
print(examples)
# # 測試OCR大數值
# image_dir = Path(r'C:\Users\david\Downloads\大數')
# for img_path in image_dir.glob('*.png'):
#     img_path = str(img_path)
#     ocr = PaddleOCR(use_angle_cls=True, lang="en")
#     ocr_result = ocr.ocr(img_path, cls=True)
#     ocr_result = ocr_result[0]
#     for data in ocr_result:
#         print(data)
#
#     image = Image.open(img_path)
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     # 顯示結果
#     plt.show()
