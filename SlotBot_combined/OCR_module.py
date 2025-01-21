from paddleocr import PaddleOCR


class Block:
    """
    Block 類別用於儲存 OCR 辨識結果中的一個區塊。

    屬性：
        - box: 區塊的座標和大小 [x, y, w, h]。
        - txt: 區塊中辨識的文字。
        - score: OCR 模型對該辨識結果的信心分數。
    """
    def __init__(self):
        self.box = []  # 區塊的座標與大小 [x, y, w, h]
        self.txt = ''  # OCR 辨識的文字
        self.score = 0  # OCR 信心分數


class FrameData:
    """
    FrameData 類別用於儲存一張圖片中所有的辨識區塊。

    屬性：
        - blocks: 包含 Block 物件的清單，代表圖片中的所有辨識結果。
    """
    def __init__(self):
        self.blocks = []  # 包含多個 Block 的清單


class OCRModule:
    """
    OCRModule 類別用於使用 PaddleOCR 對圖片進行文字辨識，並將辨識結果封裝成 FrameData。

    屬性：
        - ocr: 初始化 PaddleOCR 模組。
    """
    def __init__(self):
        # 初始化 PaddleOCR，啟用角度分類且設定語言為英語
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read_nums(self, image_path):
        """
        讀取圖片並辨識圖片中的數字。

        :param image_path: 圖片的檔案路徑。
        :return: FrameData，包含該圖片中所有辨識到的數字區塊。
        """
        # 使用 PaddleOCR 進行辨識
        result = self.ocr.ocr(image_path)
        fd = FrameData()  # 初始化 FrameData 用於儲存辨識結果

        for line in result[0]:  # 遍歷 OCR 的辨識結果
            print(line)  # 輸出每行辨識結果供檢查
            tof, number = self.clean_and_check_number(line[1][0])  # 嘗試清理並轉換為數字
            if tof:  # 如果成功轉換為數字
                bl = Block()  # 初始化 Block
                # 計算區塊的座標與大小
                x = int(line[0][0][0])  # 左上角 x 座標
                y = int(line[0][0][1])  # 左上角 y 座標
                w = int(line[0][1][0] - line[0][0][0])  # 寬度
                h = int(line[0][2][1] - line[0][1][1])  # 高度
                bl.box = [x, y, w, h]  # 設定區塊的座標和大小
                bl.txt = number  # 設定辨識的數字
                bl.score = line[1][1]  # 設定 OCR 的信心分數
                fd.blocks.append(bl)  # 將區塊加入 FrameData
        return fd  # 回傳 FrameData

    def clean_and_check_number(self, str):
        """
        清理字串並嘗試將其轉換為數字。

        :param str: OCR 辨識的字串。
        :return: (bool, float)，如果成功轉換為數字，回傳 (True, 數字)；否則回傳 (False, None)。
        """
        try:
            # 去掉字串中的逗號並嘗試轉換為浮點數
            number = float(str.replace(",", ""))
            return True, number  # 如果成功，回傳 True 和數字
        except ValueError:
            return False, None  # 如果失敗，回傳 False 和 None


if __name__ == "__main__":
    """
    主程式：測試 OCRModule 的功能。
    """
    # 初始化 OCRModule
    mod = OCRModule()
    # 設定測試圖片的路徑
    img = r'D:\git-repository\SlotGame_AutoBot\images\dragon\screenshots\base_game\dragon_round_5.png'
    # 使用 OCR 模組讀取圖片並取得辨識結果
    fd = mod.read_nums(img)

    # 輸出所有辨識區塊的資訊
    for bl in fd.blocks:
        print(bl.box)   # 輸出區塊的座標與大小
        print(bl.txt)   # 輸出區塊中的文字
        print(bl.score) # 輸出區塊的信心分數
