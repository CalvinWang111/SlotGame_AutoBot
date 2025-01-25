import os
import re
from pathlib import Path
from dotenv import load_dotenv
from ChatGPT.openai_api import OpenAiApi
from ChatGPT.prompt_format import InfoPromptFormat

class GameScenceRecognizer:
    """
    A class for recognizing information pages related to a game.
    """

    def __init__(self, game: str):
        """
        Initializes the GameScenceRecognizer instance.

        Args:
            game (str): The name of the game for which the information pages are recognized.
        """
        self.root_dir = Path(__file__).parent.parent  # Define the root directory
        self.game = game
        env_path = "../.env"  # Relative path to the .env file
        dotenv_path = Path(env_path)

        # Load environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)

        # Initialize OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api = OpenAiApi(api_key)

        self.info_prompt = InfoPromptFormat()

        self.openai_api.setting_system_role_play(self.info_prompt.system_role_play)
        self.openai_api.setting_role_play_for_image(self.info_prompt.rule_prompt)

        self.setting_rule_image()

    def setting_rule_image(self):
        """
        Reading info images to set the role description for the OpenAI API to let it understand game rules.
        """
        # Define the folder path for game information images
        info_folder_path = os.path.join(self.root_dir, "images", self.game, "info")
        info_image_paths = os.listdir(info_folder_path)
        info_image_path_list = [os.path.join(info_folder_path, info_image_path) for info_image_path in info_image_paths]


        self.openai_api.setting_image_role_play(info_image_path_list)

    def setting_base_game_image(self, game_scence_image_path: str):
        """
        Setting base game image
        """
        self.openai_api.setting_image_role_play([game_scence_image_path])

    def extract_response(self, response: str) -> bool:
        """
        Extract the answer from the response
        """
        answer = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        
        answer = False if len(answer) == 0 or answer[0] == "不是" else True
            
        explain = re.findall(r"<explain>(.*?)</explain>", response, re.DOTALL)

        return answer, explain[0]

    def recognize_game_scence(self, image_path: str) -> str:
        """
        Recognize the game board by given image
        """
        response = self.openai_api.get_gpt_response(
            image_path=image_path,
            prompt=self.info_prompt.user_prompt
        )
        answer, explain = self.extract_response(response)
        return answer, explain
    

