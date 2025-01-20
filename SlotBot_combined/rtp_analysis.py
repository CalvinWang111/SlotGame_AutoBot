from OCR_module import OCRModule, FrameData
from pathlib import Path
import cv2


class RtpTool:
    def __init__(self):
        self.ocr = OCRModule()

    def images_to_csv(self, img_paths):
        """
        讀入圖片路徑的list
        對每張圖OCR，並記錄和分類Blocks
        輸出成csv
        """
        block_set = []

        # OCR所有圖片，取得所有FrameData
        fd_list = []
        for img_path in img_paths:
            fd = self.ocr.read_nums(img_path)
            fd_list.append(fd)

        for fd in fd_list:
            for bl in fd.blocks:
                if not self.is_block_existed(bl, block_set):
                    block_set.append(bl)

        return block_set

    def is_block_existed(self, bl, st):
        for s in st:
            if self.is_same_block(s, bl):
                return True
        return False

    def is_same_block(self, bl_1, bl_2):
        x1, y1, w1, h1 = bl_1.box
        x2, y2, w2, h2 = bl_2.box
        left_pos_1  =    [x1         , y1 + h1 / 2]
        mid_pos_1   =     [x1 + w1 / 2, y1 + h1 / 2]
        right_pos_1 =   [x1 + w1    , y1 + h1 / 2]
        left_pos_2  =    [x2         , y2 + h2 / 2]
        mid_pos_2   =     [x2 + w2 / 2, y2 + h2 / 2]
        right_pos_2 =   [x2 + w2    , y2 + h2 / 2]
        tolerance = 5
        if self.is_same_position(left_pos_1, left_pos_2, tolerance) or \
                self.is_same_position(mid_pos_1, mid_pos_2, tolerance) or \
                self.is_same_position(right_pos_1, right_pos_2, tolerance):
            return True
        return False

    def is_same_position(self, point1, point2, tolerance):
        """
        判斷兩個點是否在允許的誤差範圍內被視為相同位置

        :param point1: 第一個點 [x, y]
        :param point2: 第二個點 [x, y]
        :param tolerance: 允許的誤差範圍
        :return: 如果兩點在誤差範圍內，回傳 True；否則回傳 False
        """
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance

if __name__ == "__main__":
    folder_path = Path(r"D:\git-repository\SlotGame_AutoBot\SlotBot_combined\test_images\ch")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    # 遍歷資料夾中的所有檔案
    path_list = []
    for file_path in folder_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            file_path = str(file_path)
            path_list.append(file_path)
    print(path_list)

    rtp = RtpTool()
    csv = rtp.images_to_csv(path_list)
    boxes = []
    for bl in csv:
        boxes.append(bl.box)



    # 載入圖片
    image_path = r"D:\git-repository\SlotGame_AutoBot\SlotBot_combined\test_images\ch\Screenshot_2024.12.12_02.01.28.296.png"
    image = cv2.imread(image_path)
    # 檢查是否成功載入圖片
    if image is None:
        print("無法載入圖片，請確認路徑是否正確")
        exit()
    # 取得圖片的原始尺寸
    original_height, original_width = image.shape[:2]
    # 設定顯示的最大尺寸（例如螢幕大小）
    max_width = 800
    max_height = 600
    # 計算縮放比例，確保圖片不超出最大尺寸
    scale_width = max_width / original_width
    scale_height = max_height / original_height
    scale = min(scale_width, scale_height, 1.0)  # 確保縮放比例不超過 1（即不要放大）
    # 縮放圖片
    if scale < 1.0:  # 只有在圖片尺寸大於最大尺寸時才縮小
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 根據縮放比例調整 boxes 的座標
    scaled_boxes = [[int(x * scale), int(y * scale), int(w * scale), int(h * scale)] for x, y, w, h in boxes]
    # 繪製每個 box 到圖片上
    for box in scaled_boxes:
        x, y, w, h = box
        top_left = (x, y)  # 左上角座標
        bottom_right = (x + w, y + h)  # 右下角座標
        color = (0, 255, 0)  # 矩形顏色 (綠色，BGR 格式)
        thickness = 1  # 矩形邊框厚度
        # 在圖片上畫矩形
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # 顯示圖片
    cv2.imshow("Boxes on Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


