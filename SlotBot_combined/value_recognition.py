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
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

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

    def recognize_value(self, root_dir, game, mode, image_paths, highest_confidence_images={}):
        for image_path in image_paths:
            ocr_result = self.ocr.ocr(image_path, cls=True)
            ocr_result = ocr_result[0]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            game_name = game
            json_data = {}

            frame = cv2.imread(image_path)

            if highest_confidence_images:
                btnocr = self.screenshot.spinbuttonOCR(self=self.screenshot,highest_confidence_images=highest_confidence_images, frame=frame)
                print('btnocr', btnocr)
                if not isinstance(btnocr, bool) and len(btnocr)>0:
                    mode = 'free'
                else:
                    mode = 'base'
                

            # *********************************************
            # filename = f"./json/data_{timestamp}.json"
            # 定義完整路徑
            output_dir = os.path.join(root_dir, f"output/{game_name}/numerical")
            # filename = os.path.join(output_dir, f"data_{timestamp}.json")

            if mode == 'base':
                # 還原 file 名稱
                filename = os.path.basename(image_path)
                filename = os.path.join(output_dir, filename.split('.')[0] + '.json')
            else:
                output_dir = os.path.join(output_dir, 'fg')
                print(output_dir)
                filename = os.path.basename(image_path)
                filename = re.sub(r'\d+', '', filename.split('.')[0])
                filename = os.path.join(output_dir, filename + str(btnocr[0]))


            # 檢查並創建目錄
            os.makedirs(output_dir, exist_ok=True)
            # filename = os.path.join(root_dir, f"./output/{game_name}/numerical/data_{timestamp}.json")
            # *********************************************

            for line in self.meaning_table:
                json_each_line = {}
                json_each_line['path'] = ''
                for data in ocr_result:
                    x = int(data[0][0][0])
                    y = int(data[0][0][1])
                    w = int(data[0][1][0] - data[0][0][0])
                    h = int(data[0][2][1] - data[0][1][1])
                    new_value_pos = {'roi': [x, y, w, h], 'value': data[1][0]}

                    middle = [line['roi'][0] + line['roi'][2] / 2, line['roi'][1] + line['roi'][3] / 2]
                    new_middle = [new_value_pos['roi'][0] + new_value_pos['roi'][2] / 2,
                                  new_value_pos['roi'][1] + new_value_pos['roi'][3] / 2]

                    # left side similar
                    if line['roi'][0] + self.threshold > new_value_pos['roi'][0] > line['roi'][0] - self.threshold and \
                            middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                        print(new_value_pos['value'], line['meaning'])
                        json_each_line['confidence'] = data[1][1]
                        json_each_line['contour'] = new_value_pos['roi']
                        json_each_line['value'] = new_value_pos['value']
                        json_data[line['meaning']] = json_each_line
                        break
                    # right side similar
                    elif line['roi'][0] + line['roi'][2] + self.threshold > new_value_pos['roi'][0] + \
                            new_value_pos['roi'][
                                2] > line['roi'][0] + line['roi'][2] - self.threshold and middle[1] + self.threshold > \
                            new_middle[
                                1] > middle[
                        1] - self.threshold:
                        print(new_value_pos['value'], line['meaning'])
                        json_each_line['confidence'] = data[1][1]
                        json_each_line['contour'] = new_value_pos['roi']
                        json_each_line['value'] = new_value_pos['value']
                        json_data[line['meaning']] = json_each_line

                        break
                    # middle similar
                    elif middle[0] + self.threshold > new_middle[0] > middle[0] - self.threshold and middle[
                        1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                        print(new_value_pos['value'], line['meaning'])
                        json_each_line['confidence'] = data[1][1]
                        json_each_line['contour'] = new_value_pos['roi']
                        json_each_line['value'] = new_value_pos['value']
                        json_data[line['meaning']] = json_each_line

                        break
            print('json_data', json_data)
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)
            frame = cv2.imread(image_path)
            cv2.imwrite(rf'./images/value/value+{timestamp}.png', frame)

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
