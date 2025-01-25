import base64
from pathlib import Path
from openai import OpenAI
from ChatGPT.prompt_format import PromptFormat, GetValuePromptFormat, GetSimplifiedMeaningPromptFormat

class OpenAiApi:

    def __init__(self, api_key) -> None:
        self.root_dir = Path(__file__).parent.parent
        self.openai = OpenAI(api_key=api_key)
        self.role_description = ""
        self.role_description_for_image = ""
    
    @staticmethod
    def encode_image(image_path):
        """
        encode image
        """
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def setting_system_role_play(self, role_description: str):
        """
        Sets up role-playing descriptions.

        Args:
            role_description (str): The main description of the role for the language model.
        
        Returns:
            None
        """
        self.role_description = {
            "role": "system",
            "content":[
                {
                    "type": "text",
                    "text": role_description
                }
            ]
        }
    def setting_role_play_for_image(self, role_description_for_image: str):
        """
        Sets up role-playing descriptions.

        Args:
            role_description_for_image (str): The description for the role if images are included.

        Returns:
            None
        """
        self.role_description_for_image = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": role_description_for_image
                }
            ]
        }
        
    def setting_image_role_play(self, image_path_list: list):
        """
        Sets up role-playing descriptions with image.
        Please call function setting_role_play_for_image before call this function, due to openai can't set image in "role": "system"

        Args:
            image_path_list (list): A list of file paths for images associated with the role description.

        Returns:
            None
        """
        if self.role_description_for_image == "":
            raise ValueError("Please call function setting_role_play_for_image before call this function")

        for image_path in image_path_list:
            image = self.encode_image(image_path=image_path)
            self.role_description_for_image["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            )

    def get_gpt_response(self, image_path, prompt):
        """
        get openai response
        """
        image = self.encode_image(image_path=image_path)
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                self.role_description,
                self.role_description_for_image,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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

    def get_gpt_response_with_prompt(self, prompt) -> str:
        """
        Fetches a response from the OpenAI API based on the provided prompt.

        Args:
            prompt (list): A list of messages formatted according to the OpenAI API documentation.
                        Refer to: https://platform.openai.com/docs/guides/vision#managing-images

        Returns:
            str: The content of the response message from the OpenAI API.
        """
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages = prompt,
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
            max_tokens = 500,
            temperature = 0
        )
        return response.choices[0].message.content

    def get_simplified_meaning(self, meaning_list):
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": GetSimplifiedMeaningPromptFormat.PROMPT
                        },
                        {
                            "type": "text",
                            "text": f"{meaning_list}"
                        }
                    ]
                }
            ],
            max_tokens = 500,
            temperature = 0
        )
        return response.choices[0].message.content