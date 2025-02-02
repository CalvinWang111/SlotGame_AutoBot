from OCR_module import OCRModule, FrameData
from pathlib import Path
import cv2
import csv


class RtpTool:
    def __init__(self):
        """
        初始化 RtpTool 類別，包含 OCRModule 的實例。
        """
        self.ocr = OCRModule()

    def images_to_csv(self, img_paths, output_path):
        """
        讀入圖片路徑的 list，對每張圖進行 OCR 處理，記錄和分類 Blocks，並輸出為 CSV。

        :param img_paths: 包含所有圖片路徑的 list。
        :param output_path: 輸出的 CSV 檔案路徑。
        """
        block_set = []
        # OCR 所有圖片，取得所有 FrameData
        fd_list = []
        for img_path in img_paths:
            fd = self.ocr.read_nums(img_path)
            fd_list.append(fd)

        # 取得 block set
        for fd in fd_list:
            for bl in fd.blocks:
                if not self.is_block_existed(bl, block_set):
                    block_set.append(bl)
        self.demo_block_set(block_set, img_paths[0])  # Demo block set

        block_dict = {}
        # 初始化 block_dict
        for i in range(len(block_set)):
            line_name = f'B{i}'
            block_dict[line_name] = []

        # 將文字分類到 block_dict
        for fd in fd_list:
            for i in range(len(block_set)):
                line_name = f'B{i}'
                appended = False
                for bl in fd.blocks:
                    if self.is_same_block(bl, block_set[i]):
                        block_dict[line_name].append(bl.txt)
                        appended = True
                        break
                if not appended:
                    block_dict[line_name].append('NULL')

        # 輸出為 CSV
        self.export_block_dict_to_csv(block_dict, output_path)

    def is_block_existed(self, bl, st):
        """
        檢查指定的 Block 是否已存在於 Block Set 中。

        :param bl: 欲檢查的 Block。
        :param st: Block Set。
        :return: 如果 Block 已存在，回傳 True；否則回傳 False。
        """
        for s in st:
            if self.is_same_block(s, bl):
                return True
        return False

    def is_same_block(self, bl_1, bl_2):
        """
        判斷兩個 Blocks 是否在允許的誤差範圍內被視為相同。

        :param bl_1: 第一個 Block。
        :param bl_2: 第二個 Block。
        :return: 如果兩個 Blocks 被視為相同，回傳 True；否則回傳 False。
        """
        x1, y1, w1, h1 = bl_1.box
        x2, y2, w2, h2 = bl_2.box
        left_pos_1 = [x1, y1 + h1 / 2]
        mid_pos_1 = [x1 + w1 / 2, y1 + h1 / 2]
        right_pos_1 = [x1 + w1, y1 + h1 / 2]
        left_pos_2 = [x2, y2 + h2 / 2]
        mid_pos_2 = [x2 + w2 / 2, y2 + h2 / 2]
        right_pos_2 = [x2 + w2, y2 + h2 / 2]
        tolerance = 5  # 設定誤差範圍
        if self.is_same_position(left_pos_1, left_pos_2, tolerance) or \
                self.is_same_position(mid_pos_1, mid_pos_2, tolerance) or \
                self.is_same_position(right_pos_1, right_pos_2, tolerance):
            return True
        return False

    @staticmethod
    def is_same_position(point1, point2, tolerance):
        """
        判斷兩個點是否在允許的誤差範圍內被視為相同位置。

        :param point1: 第一個點 [x, y]。
        :param point2: 第二個點 [x, y]。
        :param tolerance: 允許的誤差範圍。
        :return: 如果兩點在誤差範圍內，回傳 True；否則回傳 False。
        """
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance

    @staticmethod
    def demo_block_set(block_set, img_path):
        """
        將 Block Set 繪製在指定的圖片上，並顯示圖片。

        :param block_set: Block Set，包含所有 Blocks 的清單。
        :param img_path: 圖片路徑。
        """
        boxes = []
        for bl in block_set:
            boxes.append(bl.box)

        # 載入圖片
        image = cv2.imread(img_path)
        if image is None:
            print("無法載入圖片，請確認路徑是否正確")
            exit()

        # 調整圖片大小至螢幕範圍內
        original_height, original_width = image.shape[:2]
        max_width = 800
        max_height = 600
        scale_width = max_width / original_width
        scale_height = max_height / original_height
        scale = min(scale_width, scale_height, 1.0)
        if scale < 1.0:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 繪製 Block
        scaled_boxes = [[int(x * scale), int(y * scale), int(w * scale), int(h * scale)] for x, y, w, h in boxes]
        for box in scaled_boxes:
            x, y, w, h = box
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            color = (0, 255, 0)
            thickness = 1
            cv2.rectangle(image, top_left, bottom_right, color, thickness)

        cv2.imshow("Boxes on Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def export_block_dict_to_csv(block_dict, output_file):
        """
        將 block_dict 轉成 CSV 檔案輸出。

        :param block_dict: 包含 Block 分類資料的字典。
                           格式: { "B0": [txt1, txt2, ...], "B1": [txt3, txt4, ...], ... }。
        :param output_file: 輸出 CSV 檔案的路徑。
        """
        headers = ["Block"] + [f"Entry_{i + 1}" for i in range(len(next(iter(block_dict.values()))))]
        rows = [[block_name] + entries for block_name, entries in block_dict.items()]

        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"CSV 檔案已成功輸出到: {output_file}")


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
    rtp.images_to_csv(path_list, 'test_rtp.csv')
