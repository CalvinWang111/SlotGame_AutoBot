from paddleocr import PaddleOCR
from ChatGPT.openai_api import OpenAiApi
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from datetime import datetime
import re
import cv2


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

    # def set_position_to_meaning(self, frame_list):
    #
    def get_board_value(self, image_path):
        # Path to your image
        chat_response = self.openai_api.get_value_response(image_path)
        print(chat_response)
        chat_response = chat_response.replace("{", "").replace("}", "")
        result = chat_response.splitlines()
        count = 0
        table = [line.split(";") for line in result]

        # OCR
        ocr_result = self.ocr.ocr(image_path, cls=True)
        ocr_result = ocr_result[0]

        # compare
        for data in ocr_result:
            for meaning in table:
                # if value of OCR = value of chatgpt
                if data[1][0] == meaning[0]:
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

    def get_meaning(self):
        meaning_list = [{'position': line['roi'], 'meanings': line['meaning']} for line in self.value_pos_form]
        chat_response = self.openai_api.get_simplified_meaning(meaning_list)
        print(f'chat_response = {chat_response}')
        tuple_list = re.findall(r"<position>(.*?)</position>.*?<meaning>(.*?)</meaning>", chat_response, re.DOTALL)
        dict_list = [
            {'roi': [int(n) for n in roi.strip('[]').split(',')], 'meaning': meaning}
            for roi, meaning in tuple_list
        ]
        self.meaning_table = dict_list
        print(f'meaning = {meaning_list}')
        print(f'result = {dict_list}')

    def recognize_value(self, image_path):
        ocr_result = self.ocr.ocr(image_path, cls=True)
        ocr_result = ocr_result[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_data = {}
        filename = f"./json/data_{timestamp}.json"
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
