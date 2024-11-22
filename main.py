import os
from pathlib import Path
from ChatGPT.openai_api import OpenAiApi
from utils.env_reader import EnvData
from OCR.ocr_model import OCRmodel

def main():
    root_dir = Path(__file__).parent
    # envdata = EnvData()
    # openai_api = OpenAiApi(envdata.openai_api_key)
    ocr_model = OCRmodel()

    image_path = os.path.join(root_dir, "material", "Screenshot_13_cropped_image_6_contour_0.png")

    # response = openai_api.get_gpt_response(image_path)
    # print(response)

    ocr_model.get_free_game_btn(image_path, "result/8.png")

if __name__ == '__main__':
    main()
