import os
from pathlib import Path
from ChatGPT.openai_api import OpenAiApi
from utils.env_reader import EnvData
from OCR.ocr_model import OCRmodel

def main():
    root_dir = Path(__file__).parent
    envdata = EnvData()
    openai_api = OpenAiApi(envdata.openai_api_key)
    ocr_model = OCRmodel()

    image_path = os.path.join(root_dir, "material", "Screenshot_14.png")

    # response = openai_api.get_gpt_response(image_path)
    # print(response)


    response = ocr_model.test(image_path)
    print(response)

if __name__ == '__main__':
    main()
