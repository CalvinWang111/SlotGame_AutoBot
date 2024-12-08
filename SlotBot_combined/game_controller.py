from screenshot import GameScreenshot
import pygetwindow as gw
import pyautogui
import time
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from TemplateMatching.template_matching import test_template_matching
from pathlib import Path
import random
import json
import os

class GameController:
    @staticmethod
    def click_in_window(window_title, x_offset, y_offset, clicks=1, interval=0.0, button='left'):
        """
        Click at a specific location within a window identified by its title.
        
        Parameters:
        - window_title (str): The title of the window to interact with.
        - x_offset (int): The x offset within the window where the click should occur.
        - y_offset (int): The y offset within the window where the click should occur.
        - clicks (int): Number of clicks to perform (default is 1).
        - interval (float): Interval between clicks if multiple clicks are specified.
        - button (str): Which mouse button to click, 'left', 'right', or 'middle' (default is 'left').
        """
        try:
            # Find the window by title
            window = gw.getWindowsWithTitle(window_title)[0]  # Assumes the first match
            #window.activate()  # Bring the window to the foreground
            time.sleep(0.5)  # Wait for the window to come to the front
            
            # Get the window's position and size
            x, y = window.left, window.top
            
            # Calculate the absolute click position based on the window position
            click_x = x + x_offset
            click_y = y + y_offset
            
            # Move and click at the calculated position
            pyautogui.click(click_x, click_y, clicks=clicks, interval=interval, button=button)
            
            print(f"Clicked at ({click_x}, {click_y}) in window '{window_title}'")
            
        except IndexError:
            print(f"Window titled '{window_title}' not found.")

    def Windowcontrol(self, highest_confidence_images, classId):
        for item in highest_confidence_images.items():
            if item[0] == classId:
                x,y,w,h = item[1]['contour']
                x_offset = x + w / 2
                y_offset = y + h / 2

                print(x_offset, y_offset)
                self.click_in_window('BlueStacks App Player',x_offset, y_offset)

    def freegame_control(root_dir=Path(__file__).parent.parent,window_name='BlueStacks App Player', Snapshot=''):
        screenshot = GameScreenshot()
        vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
        sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
        #sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
        sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
        images_dir = os.path.join(root_dir, 'images')
        success_continue = False

        try:
            print('into try loop')
            screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'freegame')

            # try template
            freegame_screenshot = os.path.join(root_dir, 'images', Snapshot+'freegame.png')
            all_freegame_btn_json_path = os.path.join(root_dir, 'marquee_tool', Snapshot + '_runtime', Snapshot + '_runtime_regions.json')
            all_freegame_btn_json = json.load(open(all_freegame_btn_json_path, encoding='utf=8'))
            matched_loc = test_template_matching(freegame_screenshot, all_freegame_btn_json)
            if len(matched_loc) > 0:
                loc = random.choice(matched_loc)
                GameController.click_in_window(window_title=window_name, x_offset=loc[0] + loc[2]//2, y_offset=loc[1] + loc[3]//2)
                success_continue = True

            # try VIT model
            maskDict_path = os.path.join(images_dir, Snapshot+'freegame' + ".png")
            sam = SAMSegmentation(Snapshot=Snapshot, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)
            maskDict = sam.segment_image(maskDict_path)
                
            # Classify components
            vit = ViTRecognition(
                    Snapshot=Snapshot, 
                    maskDict=maskDict, 
                    model_path=vit_model_path
                )
            highest_confidence_images, template_folder = vit.classify_components()
            vit.output_json(template_folder=template_folder, highest_confidence_images=highest_confidence_images)
                
            # Check for specific predictions
            if any(key in [3,8,12, 13] for key in highest_confidence_images.keys()):
                key = [key in [3,8,12,13] for key in highest_confidence_images.keys()]
                print('detecting 金元寶或金幣')
                GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=key[0])
                success_continue = True
            
        except Exception as e:
            print(f"An error occurred: {e}")
        
        return success_continue