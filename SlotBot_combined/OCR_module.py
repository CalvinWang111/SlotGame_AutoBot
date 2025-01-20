from paddleocr import PaddleOCR


class Block:
    def __init__(self):
        self.box = []
        self.txt = ''
        self.score = 0


class FrameData:
    def __init__(self):
        self.blocks = []


class OCRModule:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read_nums(self, image_path):
        """
        讀入一張圖片，辨識圖片中的數字
        將辨識的數字包裝成Block，Block包含box、txt、score
        回傳list[Block]
        """
        result = self.ocr.ocr(image_path)
        blocks = []
        for line in result[0]:
            print(line)
            tof, number = self.clean_and_check_number(line[1][0])
            if tof:
                bl = Block()
                x = int(line[0][0][0])
                y = int(line[0][0][1])
                w = int(line[0][1][0] - line[0][0][0])
                h = int(line[0][2][1] - line[0][1][1])
                bl.box = [x, y, w, h]
                bl.txt = number
                bl.score = line[1][1]
                blocks.append(bl)
        return blocks

    def clean_and_check_number(self, str):
        """
        將輸入的字串轉成數字
        如果成功，回傳True和數字
        如果失敗，回傳False和None
        """
        try:
            # 去掉逗號並轉為浮點數
            number = float(str.replace(",", ""))
            return True, number  # 回傳結果和數值
        except ValueError:
            return False, None


if __name__ == "__main__":
    mod = OCRModule()
    img = r'D:\git-repository\SlotGame_AutoBot\images\dragon\screenshots\base_game\dragon_round_5.png'
    bls = mod.read_nums(img)
    for bl in bls:
        print(bl.box)
        print(bl.txt)
        print(bl.score)
