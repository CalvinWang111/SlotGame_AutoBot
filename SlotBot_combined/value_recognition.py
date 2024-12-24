from paddle.base.libpaddle.pir.ops import maximum
from paddleocr import PaddleOCR
from ChatGPT.openai_api import OpenAiApi
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from datetime import datetime
import re
import cv2
import random
import time


class ValueRecognition:
    def __init__(self):
        env_path = r"../.env"
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path, override=True)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api = OpenAiApi(self.api_key)

        self.value_pos_form = []

        self.threshold = 5
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        self.meaning_table = None

        #testing
        self.test_count = 0

    # def set_position_to_meaning(self, frame_list):
    #
    def get_board_value(self, image_path):
        # Path to your image
        chat_response = self.openai_api.get_value_response(image_path)
        #chat_response = self.openai_api.get_gpt_response(image_path)
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
            #去除數字英文以外的字元
            number_of_OCR = data[1][0]
            for char in data[1][0]:
                if not ('9' >= char >= '0' or 'Z' >= char >= 'A' or 'z' >= char >= 'a'):
                    number_of_OCR = number_of_OCR.replace(char,"")
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

    def get_meaning(self, image_paths, image_amount):
        for i in range(image_amount):
            self.get_board_value(image_paths[i])

        for i in range(len(self.value_pos_form)):
            if len(self.value_pos_form[i]['meaning']) <= 2:
                self.value_pos_form[i]['meaning'] = []
        for line in self.value_pos_form:
            print(line)
        for i in range(10):
            meaning_list = [{'position': line['roi'], 'meanings': line['meaning']} for line in self.value_pos_form]
            chat_response = self.openai_api.get_simplified_meaning(meaning_list)
            print(f'chat_response = {chat_response}')
            tuple_list = re.findall(r"<position>(.*?)</position>.*?<meaning>(.*?)</meaning>", chat_response, re.DOTALL)
            dict_list = [
                {'roi': [int(n) for n in roi.strip('[]').split(',')], 'meaning': meaning}
                for roi, meaning in tuple_list
            ]
            if self.meaning_table == None:
                self.meaning_table = dict_list
            if len(self.meaning_table) < len(dict_list):
                self.meaning_table = dict_list
        #testing
        with open(f"./json/format{self.test_count}.json", "w", encoding="utf-8") as file:
            json.dump(self.meaning_table, file, ensure_ascii=False, indent=4)

        print(f'meaning = ')
        print('\n'.join(map(str, meaning_list)))
        print(f'result = ')
        print('\n'.join(map(str, self.meaning_table)))

    def recognize_value(self, root_dir, mode, image_paths):
        for image_path in image_paths:
            ocr_result = self.ocr.ocr(image_path, cls=True)
            ocr_result = ocr_result[0]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            game_name = mode
            json_data = {}
            
            #*********************************************
            #filename = f"./json/data_{timestamp}.json"
            # 定義完整路徑
            output_dir = os.path.join(root_dir, f"output/{game_name}/numerical")
            filename = os.path.join(output_dir, f"data_{timestamp}.json")

            # 檢查並創建目錄
            os.makedirs(output_dir, exist_ok=True)
            #filename = os.path.join(root_dir, f"./output/{game_name}/numerical/data_{timestamp}.json")
            #*********************************************

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
                    elif line['roi'][0] + line['roi'][2] + self.threshold > new_value_pos['roi'][0] + new_value_pos['roi'][
                        2] > line['roi'][0] + line['roi'][2] - self.threshold and middle[1] + self.threshold > new_middle[
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
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)
            frame = cv2.imread(image_path)
            cv2.imwrite(rf'./images/value/value+{timestamp}.png', frame)

    def auto_test(self):
        folder_path = './test_images'
        files = ['./test_images/' + filename for filename in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, filename))]
        for i in range(20):
            #init
            ocr_start_time = time.time()
            self.value_pos_form = []
            self.meaning_table = None
            self.test_count = i

            sample_files = random.sample(files, min(10, len(files)))

            self.get_meaning(sample_files, 10)
            ocr_total_run_time = time.time() - ocr_start_time
            print(f'round: {i} ocr_total_run_time = {ocr_total_run_time}')


#valuerec = ValueRecognition()
#valuerec.auto_test()