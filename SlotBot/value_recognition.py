from paddleocr import PaddleOCR
from ChatGPT.openai_api import OpenAiApi
from dotenv import load_dotenv
import os
from pathlib import Path


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

    # def set_position_to_meaning(self, frame_list):
    #
    def get_board_value(self, image_path):
        # Path to your image
        chat_response = self.openai_api.get_value_response(image_path)

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
                        top_left = line['roi'][0]
                        bottom_right = line['roi'][2]
                        middle = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
                        new_middle = [(new_value_pos['roi'][0][0] + new_value_pos['roi'][2][0]) / 2,
                                      (new_value_pos['roi'][0][1] + new_value_pos['roi'][2][1]) / 2]

                        # left side similar
                        if top_left[0] + self.threshold > new_value_pos['roi'][0][0] > top_left[0] - self.threshold and \
                                middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                            line['value'].append(new_value_pos['value'][0])
                            line['meaning'].append(new_value_pos['meaning'][0])
                            found = True
                            break
                        # right side similar
                        elif bottom_right[0] + self.threshold > new_value_pos['roi'][2][0] > bottom_right[
                            0] - self.threshold and middle[1] + self.threshold > new_middle[1] > middle[
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
        for line in self.value_pos_form:
            meaning_list = line['meaning']
            chat_response = self.openai_api.get_simplified_meaning(meaning_list)
            print(f'meaning = {meaning_list}')
            print(f'response = {chat_response}')
            line['meaning'] = f'response = {chat_response}'

    def recognize_value(self, image_path):
        ocr_result = self.ocr.ocr(image_path, cls=True)
        ocr_result = ocr_result[0]

        for data in ocr_result:
            for line in self.value_pos_form:
                top_left = line['roi'][0]
                bottom_right = line['roi'][2]
                middle = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
                new_middle = [(data[0][0][0] + data[0][2][0]) / 2,
                              (data[0][0][1] + data[0][2][1]) / 2]

                # left side similar
                if top_left[0] + self.threshold > data[0][0][0] > top_left[0] - self.threshold and \
                        middle[1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                    print(data[1][0], line['meaning'])
                    break
                # right side similar
                elif bottom_right[0] + self.threshold > data[0][2][0] > bottom_right[
                    0] - self.threshold and middle[1] + self.threshold > new_middle[1] > middle[
                    1] - self.threshold:
                    print(data[1][0], line['meaning'])
                    break
                # middle similar
                elif middle[0] + self.threshold > new_middle[0] > middle[0] - self.threshold and middle[
                    1] + self.threshold > new_middle[1] > middle[1] - self.threshold:
                    print(data[1][0], line['meaning'])
                    break

