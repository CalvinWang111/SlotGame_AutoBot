from paddle.base.libpaddle.pir.ops import maximum
from paddleocr import PaddleOCR
from ChatGPT.openai_api import OpenAiApi
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from datetime import datetime
from stopping_detection import StoppingFrameCapture
from screenshot import GameScreenshot
import re
import cv2
import random
import time
import re
from PIL import Image


class ValueRecognition:
    def __init__(self):
        env_path = r"../.env"
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path, override=True)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api = OpenAiApi(self.api_key)
        self.screenshot = GameScreenshot()

        self.value_pos_form = []

        self.threshold = 5
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",show_log=False)

        self.meaning_table = None

        # testing
        self.test_count = 0

        self.window_size = {}

    # def set_position_to_meaning(self, frame_list):
    #
    def get_board_value(self, image_path):
        # Path to your image
        chat_response = self.openai_api.get_value_response(image_path)
        # chat_response = self.openai_api.get_gpt_response(image_path)
        print(chat_response)
        tuple_list = re.findall(r"<number>(.*?)</number>.*?<meaning>(.*?)</meaning>", chat_response, re.DOTALL)
        table = [
            [number, meaning]
            for number, meaning in tuple_list
        ]

        # OCR
        ocr_result = self.ocr.ocr(image_path, cls=True)
        ocr_result = ocr_result[0]

        # compare
        for data in ocr_result:
            # 去除數字英文以外的字元
            number_of_OCR = data[1][0]
            for char in data[1][0]:
                if not ('9' >= char >= '0' or 'Z' >= char >= 'A' or 'z' >= char >= 'a'):
                    number_of_OCR = number_of_OCR.replace(char, "")
            for meaning in table:
                # 去除數字英文以外的字元
                number_of_GPT = meaning[0]
                for char in meaning[0]:
                    if not ('9' >= char >= '0' or 'Z' >= char >= 'A' or 'z' >= char >= 'a'):
                        number_of_GPT = number_of_GPT.replace(char, "")
                # if value of OCR = value of chatgpt
                if number_of_OCR == number_of_GPT:
                    x = int(data[0][0][0])
                    y = int(data[0][0][1])
                    w = int(data[0][1][0] - data[0][0][0])
                    h = int(data[0][2][1] - data[0][1][1])
                    new_value_pos = {'roi': [x, y, w, h], 'value': [data[1][0]], 'meaning': [meaning[1]]}
                    found = False
                    for line in self.value_pos_form:
                        middle = [line['roi'][0] + line['roi'][2] / 2, line['roi'][1] + line['roi'][3] / 2]
                        new_middle = [new_value_pos['roi'][0] + new_value_pos['roi'][2] / 2,
                                      new_value_pos['roi'][1] + new_value_pos['roi'][3] / 2]

                        # left side similar
                        if line['roi'][0] + self.threshold > new_value_pos['roi'][0] > line['roi'][
                            0] - self.threshold and \
                                middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                            line['value'].append(new_value_pos['value'][0])
                            line['meaning'].append(new_value_pos['meaning'][0])
                            found = True
                            break
                        # right side similar
                        elif line['roi'][0] + line['roi'][2] + self.threshold > new_value_pos['roi'][0] + \
                                new_value_pos['roi'][2] > line['roi'][
                            0] + line['roi'][2] - self.threshold and middle[1] + self.threshold > new_middle[1] > \
                                middle[
                                    1] - self.threshold:
                            line['value'].append(new_value_pos['value'][0])
                            line['meaning'].append(new_value_pos['meaning'][0])
                            found = True
                            break
                        # middle similar
                        elif middle[0] + self.threshold > new_middle[0] > middle[0] - self.threshold and middle[
                            1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                            line['value'].append(new_value_pos['value'][0])
                            line['meaning'].append(new_value_pos['meaning'][0])
                            found = True
                            break

                    if not found:
                        self.value_pos_form.append(new_value_pos)

        for line in self.value_pos_form:
            print(line)

    def get_meaning(self, root_dir, game, mode, image_paths, image_amount):
        """
            整理數值意義與位置對應

            Args:
                root_dir(str): 專案根目錄
                game(str): 遊戲名稱 ex:golden
                mode(str): 模式名稱 ex:base
                image_paths(list[str]): 存圖片路徑的list
                image_amount(int): 要用幾張list的圖片
            Returns:
                None
            Raises:
                None
        """
        json_file_name = f"{game}_{mode}.json"
        file_path = rf"{root_dir}\json\{json_file_name}"
        print(f'file_path = {file_path}')

        # 確保中間資料夾存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 獲取圖片大小 (寬, 高)
        image = Image.open(image_paths[0])
        width, height = image.size
        self.window_size = {'window_size': [width, height]}

        # 檢查檔案是否存在
        if os.path.exists(file_path):
            # 檔案存在，讀取 JSON
            with open(file_path, "r+", encoding="utf-8") as file:
                json_data = json.load(file)
                print("已讀取檔案內容：", json_data)
                # 分離 window_size 和其他物件
                json_window_size = None
                json_meaning_table = []
                for item in json_data:
                    if "window_size" in item:
                        json_window_size = item["window_size"]
                    else:
                        json_meaning_table.append(item)
                if self.window_size['window_size'] == json_window_size:
                    self.meaning_table = json_meaning_table
                else:
                    # 計算縮放比例
                    scale_x = self.window_size['window_size'][0] / json_window_size[0]
                    scale_y = self.window_size['window_size'][1] / json_window_size[1]
                    for obj in json_meaning_table:
                        obj["roi"] = [
                            int(obj["roi"][0] * scale_x),  # x
                            int(obj["roi"][1] * scale_y),  # y
                            int(obj["roi"][2] * scale_x),  # width
                            int(obj["roi"][3] * scale_y)  # height
                        ]
                    self.meaning_table = json_meaning_table
                    scaled_data = self.meaning_table + [self.window_size]
                    # 將資料寫入 JSON 檔案
                    file.seek(0)
                    json.dump(scaled_data, file, ensure_ascii=False, indent=4)
                    file.truncate()
                    print(f"縮放後的資料已成功寫入檔案：{file_path}")
        else:
            # 檔案不存在，創建新檔案
            with open(file_path, "w", encoding="utf-8") as file:
                # 初始化
                self.value_pos_form = []

                for i in range(image_amount):
                    self.get_board_value(image_paths[i])
                for i in range(len(self.value_pos_form)):
                    if len(self.value_pos_form[i]['meaning']) <= 2:
                        self.value_pos_form[i]['meaning'] = []
                for line in self.value_pos_form:
                    print(line)
                for i in range(10):
                    meaning_list = [{'position': line['roi'], 'meanings': line['meaning']} for line in
                                    self.value_pos_form]
                    chat_response = self.openai_api.get_simplified_meaning(meaning_list)
                    print(f'chat_response = {chat_response}')
                    tuple_list = re.findall(r"<position>(.*?)</position>.*?<meaning>(.*?)</meaning>", chat_response,
                                            re.DOTALL)
                    dict_list = [
                        {'roi': [int(n) for n in roi.strip('[]').split(',')], 'meaning': meaning}
                        for roi, meaning in tuple_list
                    ]
                    if self.meaning_table is None:
                        self.meaning_table = dict_list
                    if len(self.meaning_table) < len(dict_list):
                        self.meaning_table = dict_list

                data = self.meaning_table + [self.window_size]
                json.dump(data, file, ensure_ascii=False, indent=4)
                print("檔案不存在，已創建新檔案，內容為：", data)


    def get_image_files(self, directory):
        """讀取所有符合 round_XX-YY 命名規則的檔案"""
        pattern = re.compile(r"(.+)_round_(\d+)-(\d+)\.(\w+)$")
        image_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                match = pattern.match(file)
                if match:
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        return image_files
    

    def shift_round_numbers(self, paths, start_round, stop_frame_old, btnocr_records):
        """
        進行 round shift：
        1. 先將 round_(start_round+1) 及以後的 rounds 往後移一 round，確保空出目標 round。
        2. 再將 round_(start_round)-stop_frame_old ~ round_(start_round)-max 變成 round_(start_round+1)-0 ~ round_(start_round+1)-N。
        3. 檢查檔案是否存在，如目標名稱已存在，則繼續往後推。
        4. 變更後更新 btnocr_records，確保檔案名稱同步。
        """
        pattern = re.compile(r"(.+)_round_(\d+)(?:-(\d+))?\.(\w+)$")
        files_to_rename = []
        later_rounds = []
        rename_map = {}  # 存放 {舊檔名: 新檔名}，用於更新 btnocr_records

        # **分類文件**
        for file_path in paths:
            filename = os.path.basename(file_path)
            match = pattern.match(filename)
            if match:
                game_name, round_num, stop_frame, ext = match.groups()
                stop_frame = int(stop_frame) if stop_frame is not None else None
                round_num, stop_frame_old = int(round_num), int(stop_frame_old)

                if round_num >= start_round + 1:
                    later_rounds.append((file_path, game_name, round_num, stop_frame, ext))
                elif round_num >= start_round and stop_frame is None:  # ✅ 把 dragon_round_13 這類加入
                    later_rounds.append((file_path, game_name, round_num, -1, ext))  # -1 代表無停輪數值
                elif round_num == start_round and stop_frame >= stop_frame_old:
                    files_to_rename.append((file_path, game_name, round_num, stop_frame, ext))

        # **1️⃣ 先處理所有 round_13+，確保有空位**
        later_rounds.sort(key=lambda x: (-x[2], x[3] if x[3] is not None else -1))  # ✅ None 變成 -1，確保能排序
        
        for old_path, game_name, round_num, stop_frame, ext in later_rounds:
            new_round = round_num + 1
            new_name = f"{game_name}_round_{new_round}{f'-{stop_frame}' if stop_frame is not None else ''}.{ext}"
            new_path = os.path.join(os.path.dirname(old_path), new_name)

            os.rename(old_path, new_path)
            rename_map[old_path] = new_path  # **記錄變更**
            print(f"🔄 {old_path} -> {new_path}")

        # **2️⃣ 重新讀取檔案，並處理 round_12-3 ~ round_12-max**
        paths = self.get_image_files(os.path.dirname(paths[0]))  # **重新讀取檔案列表**
        files_to_rename.sort(key=lambda x: x[3])  # stop_frame 升序排序

        for i, (old_path, game_name, round_num, stop_frame, ext) in enumerate(files_to_rename):
            new_round = round_num + 1
            new_stop_frame = i
            new_name = f"{game_name}_round_{new_round}-{new_stop_frame}.{ext}"
            new_path = os.path.join(os.path.dirname(old_path), new_name)

            os.rename(old_path, new_path)
            rename_map[old_path] = new_path  # **記錄變更**
            print(f"✅ {old_path} -> {new_path}")

        # **3️⃣ 更新 btnocr_records**
        new_btnocr_records = []
        for file_path, btnocr in btnocr_records:
            new_path = rename_map.get(file_path, file_path)  # 若變更則更新，未變更則保持原樣
            new_btnocr_records.append((new_path, btnocr))

        print("✅ btnocr_records 已更新！")
        return new_btnocr_records  # **返回更新後的 btnocr_records**

    def shift_round_tofront(self, image_paths, start_round):
        """
        當某些圖片合併時，調整後續 round 讓編號連續。
        - start_round: 從哪個 round 之後開始重新編號
        """
        pattern = re.compile(r"(.+)_round_(\d+)-(\d+)\.(\w+)$")  # ex: dragon_round_24-1.png
        shift_map = {}  # key: 舊 round -> value: 新 round
        new_round = start_round

        # ✅ 修正排序，確保 round 和 stop_frame 正確順序
        image_paths.sort(key=lambda p: tuple(map(int, re.findall(r"_(\d+)-(\d+)\.", p)[0])))

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            match = pattern.match(filename)

            if not match:
                continue  # 不符合格式跳過

            game_name, round_num, stop_frame, ext = match.groups()
            round_num = int(round_num)
            stop_frame = int(stop_frame)

            if round_num < start_round:
                continue  # 這些 round 不用動

            # ✅ 確保 new_round 連續不跳號
            if round_num not in shift_map:
                shift_map[round_num] = new_round
                new_round += 1

            new_round_num = shift_map[round_num]
            new_filename = f"{game_name}_round_{new_round_num}-{stop_frame}.{ext}"
            new_path = os.path.join(os.path.dirname(image_path), new_filename)

            os.rename(image_path, new_path)
            print(f"✅ 編號前移: {image_path} -> {new_path}")


    def merge_rounds(self, btnocr_records):
        print(btnocr_records)
        """
        找出相同 btnocr[0] 但 round 不同的圖片，合併至相同 round，確保 stop_frame 連續，
        並且讓後續所有 round 順序不亂。
        """

        # ✅ 按照 btnocr[0] 分組
        btnocr_groups = {}
        pattern = re.compile(r"(.+)_round_(\d+)-(\d+)\.(\w+)$")  # ex: dragon_round_24-1.png
        all_image_paths = [image_path for image_path, _ in btnocr_records]

        for image_path, btnocr in btnocr_records:
            btn_value = btnocr[0]  # 取出 btnocr[0] 來分組

            if btn_value not in btnocr_groups:
                btnocr_groups[btn_value] = []

            btnocr_groups[btn_value].append((image_path, btnocr))
 
        for btn_value, images in btnocr_groups.items():
            images.sort(key=lambda x: x[0])  # 依照檔名順序排序

            base_round = None  # 目標 round
            max_stop_frame = -1  # 記錄該 round 內最大 stop_frame

            for i, (image_path, btnocr) in enumerate(images):
                filename = os.path.basename(image_path)
                match = pattern.match(filename)

                if not match:
                    continue  # 不符合格式就跳過

                game_name, round_num, stop_frame, ext = match.groups()
                round_num = int(round_num)
                stop_frame = int(stop_frame)

                if base_round is None:
                    base_round = round_num  # 設定為第一個 round
                    max_stop_frame = stop_frame  # 更新最大 stop_frame
                    continue

                if round_num != base_round:
                    # 發現不同 round，合併
                    # 先找出 base_round 內的最大 stop_frame
                    existing_stop_frames = [
                        int(re.search(rf"{game_name}_round_{base_round}-(\d+)\.{ext}", os.path.basename(p)).group(1))
                        for p, _ in images if re.search(rf"{game_name}_round_{base_round}-(\d+)\.{ext}", os.path.basename(p))
                    ]
                    max_stop_frame = max(existing_stop_frames, default=-1) + 1  # 找不到時從 0 開始

                    # 生成新文件名
                    new_filename = f"{game_name}_round_{base_round}-{max_stop_frame}.{ext}"
                    new_path = os.path.join(os.path.dirname(image_path), new_filename)

                    # 確保不會重命名為已存在的檔案名稱
                    while os.path.exists(new_path):
                        max_stop_frame += 1
                        new_filename = f"{game_name}_round_{base_round}-{max_stop_frame}.{ext}"
                        new_path = os.path.join(os.path.dirname(image_path), new_filename)

                    # 重新命名
                    os.rename(image_path, new_path)
                    images[i] = (new_path, btnocr)  # 更新記錄

                    print(f"✅ 檔案合併: {image_path} -> {new_path}")
        # ✅ 確保所有 round 編號是連續的
        #self.shift_round_tofront(all_image_paths, base_round + 1)

    print("✅ 所有 round 已對齊且編號連續！")


    def extract_round_number(self, filename):
        """從文件名提取 round 數字"""
        match = re.search(r'round_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    def extract_stop_frame(self, filename):
        """從文件名提取 stop_frame 數字"""
        match = re.search(r'round_\d+-(\d+)', filename)
        return int(match.group(1)) if match else -1  # 若無 stop_frame，則為 -1

    def ensure_continuous_rounds(self, image_paths):
        """確保 round 數字是連續的，並重新命名檔案"""
        pattern = re.compile(r"(.*)_round_(\d+)-(\d+)\.(\w+)")
        
        round_data = []
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            match = pattern.match(filename)
            if match:
                game_name, round_num, stop_frame, ext = match.groups()
                round_data.append((int(round_num), int(stop_frame), image_path, game_name, ext))

        # 按 round 及 stop_frame 排序
        round_data.sort()

        # 重新分配 round 數字
        round_mapping = {}
        new_round_num = 0
        renamed_paths = []

        for old_round, stop_frame, old_path, game_name, ext in round_data:
            if old_round not in round_mapping:
                round_mapping[old_round] = new_round_num
                new_round_num += 1  # 確保 round 連續

            new_round = round_mapping[old_round]
            new_filename = f"{game_name}_round_{new_round}-{stop_frame}.{ext}"
            new_path = os.path.join(os.path.dirname(old_path), new_filename)

            os.rename(old_path, new_path)
            renamed_paths.append(new_path)
            print(f"✅ {old_path} -> {new_path}")

        print("✅ 所有 round 已連續編號！")
        return renamed_paths  # 回傳新路徑

    def json_output(self, output_dir, image_paths):
        """OCR 辨識並輸出 JSON"""

        # 先確保 round 連續
        image_paths = self.ensure_continuous_rounds(image_paths)

        for image_path in image_paths:
            filename = os.path.basename(image_path)

            # 進行 OCR
            ocr_result = self.ocr.ocr(image_path, cls=True)
            ocr_result = ocr_result[0]
            json_data = {}

            # ✅ **確保輸出目錄存在**
            os.makedirs(output_dir, exist_ok=True)

            for line in self.meaning_table:
                json_each_line = {'path': ''}
                for data in ocr_result:
                    x = int(data[0][0][0])
                    y = int(data[0][0][1])
                    w = int(data[0][1][0] - data[0][0][0])
                    h = int(data[0][2][1] - data[0][1][1])
                    new_value_pos = {'roi': [x, y, w, h], 'value': data[1][0]}

                    middle = [line['roi'][0] + line['roi'][2] / 2, line['roi'][1] + line['roi'][3] / 2]
                    new_middle = [new_value_pos['roi'][0] + new_value_pos['roi'][2] / 2,
                                new_value_pos['roi'][1] + new_value_pos['roi'][3] / 2]

                    if line['roi'][0] + self.threshold > new_value_pos['roi'][0] > line['roi'][0] - self.threshold and \
                            middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                        json_each_line.update({
                            'confidence': data[1][1],
                            'contour': new_value_pos['roi'],
                            'value': new_value_pos['value']
                        })
                        json_data[line['meaning']] = json_each_line
                        break

                    elif line['roi'][0] + line['roi'][2] + self.threshold > new_value_pos['roi'][0] + \
                            new_value_pos['roi'][2] > line['roi'][0] + line['roi'][2] - self.threshold and \
                        middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                        json_each_line.update({
                            'confidence': data[1][1],
                            'contour': new_value_pos['roi'],
                            'value': new_value_pos['value']
                        })
                        json_data[line['meaning']] = json_each_line
                        break

                    elif middle[0] + self.threshold > new_middle[0] > middle[0] - self.threshold and \
                            middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                        json_each_line.update({
                            'confidence': data[1][1],
                            'contour': new_value_pos['roi'],
                            'value': new_value_pos['value']
                        })
                        json_data[line['meaning']] = json_each_line
                        break

            # ✅ **確保 `filename` 最終格式正確**
            filename_clean = os.path.splitext(filename)[0]
            json_filename = os.path.join(output_dir, filename_clean + '.json')

            print('filename', filename)
            print('filename clean', filename_clean)
            print('json_filename', json_filename)
            print('json_data', json_data)

            with open(json_filename, "w", encoding="utf-8") as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)

    
    def recognize_value(self, root_dir, game, mode, image_paths, highest_confidence_images={}):


        index = 0  # ✅ 記錄當前處理的位置
        btnocr_records = []  # ✅ 存放按鈕的 OCR 記錄 [[image_path, btnocr], ...]
        last_btnocr_first = None  # ✅ 用來判斷是否進入新的一輪 fg

        while index < len(image_paths):
            # ✅ **更新 image_paths**
            image_dir = Path(f'./images/{game}/screenshots/base_game2')
            
            image_paths = sorted(
                [os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                key=lambda x: (self.extract_round_number(os.path.basename(x)), self.extract_stop_frame(os.path.basename(x)))
            )
            print("✅ 已更新最新的 image_paths")
            

            image_path = image_paths[index]  # 取得當前圖片
            filename = os.path.basename(image_path)
            
            ocr_result = self.ocr.ocr(image_path, cls=True)
            ocr_result = ocr_result[0]
            ocr_switch = True
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            game_name = game
            json_data = {}

            frame = cv2.imread(image_path)

            # ✅ **檢查 btnocr，決定 mode**
            btnocr = None
            if highest_confidence_images:
                btnocr = self.screenshot.spinbuttonOCR(self=self.screenshot, highest_confidence_images=highest_confidence_images, frame=frame)
                print('btnocr', btnocr)
                mode = 'free' if isinstance(btnocr, list) and len(btnocr) == 2 else 'base'

            output_dir = os.path.join(root_dir, f"output/{game_name}/numerical")

            # ✅ **檔名匹配 `dragon_round_xxx`**
            pattern = re.compile(r"(.+)_round_(\d+)(?:-(\d+))?\.(\w+)$")
            match = pattern.match(filename)

            if match:
                game_name_old, round_num_old, stop_frame_old, ext_old = match.groups()
                round_num_old = int(round_num_old)
                print('正在運行round', round_num_old, '停輪偵', stop_frame_old)

                if mode == 'free' and stop_frame_old is None:
                    # ✅ **找出該 round 的最大 stop_frame**
                    same_round_files = []
                    for file_path in image_paths:
                        file_name = os.path.basename(file_path)
                        match = pattern.match(file_name)
                        if match:
                            _, round_num, stop_frame, _ = match.groups()
                            if round_num and int(round_num) == round_num_old:
                                stop_frame = int(stop_frame) if stop_frame else -1
                                same_round_files.append((file_path, stop_frame))

                    # 找最大停輪幀 stop_frame
                    max_stop_frame = max([sf[1] for sf in same_round_files]) if same_round_files else -1
                    new_stop_frame = max_stop_frame + 1

                    # ✅ **重新命名檔案**
                    new_filename = f"{game_name_old}_round_{round_num_old}-{new_stop_frame}.{ext_old}"
                    new_path = os.path.join(os.path.dirname(image_path), new_filename)
                    os.rename(image_path, new_path)

                    print(f"✅ Renamed {image_path} -> {new_path}")

                    #更換path後當輪不進行btnocr
                    ocr_switch = False
                    print('ocr switch關閉', ocr_switch)

                    # ✅ **更新 `image_path` 和 `filename`**
                    image_path = new_path
                    filename = new_filename

                    # ✅ **更新 image_paths**
                    image_dir = Path(f'./images/{game}/screenshots/base_game')
                    image_paths = sorted(
                        [os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                        key=lambda x: (self.extract_round_number(os.path.basename(x)), self.extract_stop_frame(os.path.basename(x)))
                    )
                    #print("✅ 已更新最新的 image_paths:", image_paths)

                    '''
                    # ✅ **確保從新 image_paths 中找到對應的 index**
                    if new_path in image_paths:
                        index = image_paths.index(new_path) + 1  # 找到新圖片的位置，繼續下一張
                    else:
                        index = 0  # 若檔名變更，從新列表開始
                    '''
                    if index >= len(image_paths):
                        break  # 若所有圖片處理完畢，結束迴圈

                    continue
                elif mode == 'base':
                    # ✅ **刪除符合 `dragon_round_xxx` 格式但沒有 `-數值` 的檔案**
                    if stop_frame_old is None:
                        print(f"🗑️ Deleting {image_path}")
                        os.remove(image_path)
                        
                        # ✅ **更新 image_paths**
                        image_paths.remove(image_path)

                        image_paths = sorted(
                            [os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                            key=lambda x: (self.extract_round_number(os.path.basename(x)), self.extract_stop_frame(os.path.basename(x)))
                        )

                        if index >= len(image_paths):
                            break  # 若所有圖片處理完畢，結束迴圈

                        continue    

            # ✅ **記錄 btnocr**
            if mode=='free' and btnocr and ocr_switch:
                btnocr_records.append([image_path, btnocr])

                if btnocr[0] == 4:
                    print('檢查fg4')

                if last_btnocr_first is not None and btnocr[0] != last_btnocr_first:
                    # ✅ **只有當 btnocr[0] 變化且不等於上一輪的數字時，才進入新的一輪**
                    if last_btnocr_first is None or btnocr[0] < last_btnocr_first:
                        print("🔄 btnocr 變更，進入新的一輪")
                        # ✅ **進行 round 數後移**
                        btnocr_records = self.shift_round_numbers(image_paths, round_num_old, stop_frame_old, btnocr_records)
                        # ✅ **修改當前 round**
                        new_round_num = round_num_old + 1
                        new_filename = f"{game_name_old}_round_{new_round_num}-0.{ext_old}"
                        new_path = os.path.join(os.path.dirname(image_path), new_filename)
                        # os.rename(image_path, new_path)
                        filename = new_filename
                        print(f"✅ {image_path} -> {new_path} (新 fg 輪)")

                        # ✅ **更新 `image_path`**
                        image_path = new_path

                last_btnocr_first = btnocr[0]  # 更新 btnocr 記錄

            index += 1  # ✅ **只在沒有 rename 時才往前進**
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        #檢查fg 狀態round是否需要合併
        self.merge_rounds(btnocr_records=btnocr_records)
        self.json_output(output_dir=output_dir, image_paths=image_paths)





    def auto_test(self):
        #folder_path = './test_images/ch'
        GAME = 'dragon'
        folder_path = Path(f'./images/{GAME}/screenshots/base_game')
        files = [os.path.join(folder_path, file)for file in os.listdir(folder_path)]

        '''
        files = ['./test_images/ch' + filename for filename in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, filename))]
        '''
        
        root_dir = Path(__file__).parent.parent
        MODE = 'base'
        GAME = 'golden'
        for i in range(20):
            # init
            ocr_start_time = time.time()
            self.value_pos_form = []
            self.meaning_table = None
            self.test_count = i

            sample_files = random.sample(files, min(10, len(files)))

            self.get_meaning(root_dir, GAME, MODE, sample_files, 10)
            ocr_total_run_time = time.time() - ocr_start_time
            print(f'round: {i} ocr_total_run_time = {ocr_total_run_time}')

    # Function to load a JSON file
    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Function to merge JSON files based on value and ensure all keys are included
    def merge_json_files(self, input_folder):
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

        # Group files by their base name (e.g., dragon_round_0)
        file_groups = {}
        for file_name in json_files:
            base_name = file_name.split('-')[0]  # Extract the part before the first '-'
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_name)

        # Define the output folder
        output_folder = os.path.join(input_folder, "after")
        os.makedirs(output_folder, exist_ok=True)

        for base_name, files in file_groups.items():
            # Separate files into primary and secondary based on the round_x-y format
            secondary_files = [f for f in files if '-' in f]

            if secondary_files:
                # Find the file with the highest y index
                highest_y_file = max(secondary_files, key=lambda f: int(f.split('-')[1].split('.')[0]))

                # Collect all keys and merge data
                all_values = set()
                merged_data = []

                for file_name in secondary_files:
                    file_path = os.path.join(input_folder, file_name)
                    file_data = self.load_json(file_path)

                    for key, entry in file_data.items():
                        # Ensure all keys and values are preserved
                        if 'path' in entry and 'confidence' in entry and 'contour' in entry and 'value' in entry:
                            merged_data.append(entry)
                            all_values.add(entry['value'])

                # Deduplicate data while ensuring all keys and values are included
                value_map = {}
                for entry in merged_data:
                    value_key = entry['value']
                    if value_key not in value_map:
                        value_map[value_key] = entry

                # Save merged data
                output_data = list(value_map.values())
                output_path = os.path.join(output_folder, f"{base_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)

                print(f"Merged files for {base_name} saved to {output_path}")

            else:
                # Rename files that have no secondary versions
                for file_name in files:
                    if file_name.endswith('-0.json'):
                        old_path = os.path.join(input_folder, file_name)
                        new_name = file_name.replace('-0', '')
                        new_path = os.path.join(output_folder, new_name)
                        os.rename(old_path, new_path)
                        print(f"Renamed {file_name} to {new_name}")


if __name__ == "__main__":
    valuerec = ValueRecognition()
    #valuerec.auto_test()

    # Input folder path
    input_folder = r'C:\Users\13514\SlotGame_AutoBot\output\dragon\numerical'  # Replace with your folder path

    # Call the merge function
    valuerec.merge_json_files(input_folder)
