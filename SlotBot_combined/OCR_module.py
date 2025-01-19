from paddleocr import PaddleOCR


class OCR_module:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read_nums(self, image_path):
        result = self.ocr.ocr(image_path)

        boxes = []
        txts = []
        scores = []
        for line in result[0]:
            print(line)
            tof, number = self.clean_and_check_number(line[1][0])
            if tof:
                boxes.append(line[0])
                txts.append(number)
                scores.append(line[1][1])
        return boxes, txts, scores

    def clean_and_check_number(self, str):
        try:
            # 去掉逗號並轉為浮點數
            number = float(str.replace(",", ""))
            return True, number  # 回傳結果和數值
        except ValueError:
            return False, None


if __name__ == "__main__":
    mod = OCR_module()
    img = r'D:\git-repository\SlotGame_AutoBot\images\dragon\screenshots\base_game\dragon_round_0.png'
    boxes, txts, scores = mod.read_nums(img)
    print(txts)
