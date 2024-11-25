import base64
from pathlib import Path
from openai import OpenAI
from ChatGPT.prompt_format import PromptFormat, GetValuePromptFormat

class OpenAiApi:

    def __init__(self, api_key) -> None:
        self.root_dir = Path(__file__).parent.parent
        self.openai = OpenAI(api_key=api_key)

    @staticmethod
    def __encode_image(image_path):
        """
        encode image
        """
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_gpt_response(self, image_path):
        """
        get openai response
        """
        image = self.__encode_image(image_path)
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PromptFormat.PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens = 300,
            temperature = 0.7
        )
        return response.choices[0].message.content

    def get_value_response(self, image_path):
        """
        get openai response
        """
        image = self.__encode_image(image_path)
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": GetValuePromptFormat.PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens = 300,
            temperature = 0.7
        )
        return response.choices[0].message.content
