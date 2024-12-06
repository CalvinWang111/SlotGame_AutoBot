import os
import time
import json
import random
from datetime import datetime
from pathlib import Path
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from TemplatMatching.template_matching import test_template_matching
from value_recognition import ValueRecognition
from PIL import Image


def main():
    # 初始化模組
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'GoldenHoYeah'
    intensity_threshold = 10
    spin_round = 11
    root_dir = Path(__file__).parent.parent
    print('rootdir', root_dir)

    vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
    sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    # sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
    sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
    images_dir = os.path.join(root_dir, 'images')

    print('sam_model_path', sam_model_path)
    print('sam_model_cfg', sam_model_cfg)

    sam = SAMSegmentation(Snapshot=Snapshot, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)
    valuerec = ValueRecognition()

    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)

    # 2. SAM 分割
    maskDict = sam.segment_image(os.path.join(root_dir, 'images', Snapshot + ".png"))

    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=vit_model_path)
    highest_confidence_images, template_folder = vit.classify_components()
    vit.output_json(template_folder=template_folder, highest_confidence_images=highest_confidence_images)

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + '_intialshot')

    intialshot_path = os.path.join(images_dir, Snapshot + "_intialshot.png")
    intial_avg_intensities = screenshot.clickable(snapshot_path=intialshot_path,
                                                  highest_confidence_images=highest_confidence_images)

    ocr_start_time = time.time()
    for i in range(spin_round):
        GameController.Windowcontrol(GameController, highest_confidence_images=highest_confidence_images, classId=10)
        print('spin round : ', i)
        if i == 0:
            valuerec.get_board_value(intialshot_path)
        elif i <= 9:
            valuerec.get_board_value(snapshot_path)
        else:
            valuerec.recognize_value(snapshot_path)

        if i == 9:
            valuerec.get_meaning()
            ocr_total_run_time = time.time() - ocr_start_time
            print(f'ocr_total_run_time = {ocr_total_run_time}')
        time.sleep(3)
        start_time = time.time()

        # 設定初始值，以此進入while迴圈
        avg_intensities = {
            class_id: [value + intensity_threshold for value in intensities]
            for class_id, intensities in intial_avg_intensities.items()
        }

        while (screenshot.intensity_check(intial_avg_intensities, avg_intensities, intensity_threshold)):

            elapsed_time = time.time() - start_time
            print('elapsed time', elapsed_time)
            screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + '_runtime')

            snapshot_path = os.path.join(images_dir, Snapshot + "_runtime.png")
            avg_intensities = screenshot.clickable(snapshot_path=snapshot_path,
                                                   highest_confidence_images=highest_confidence_images)
            print('waiting')

            if elapsed_time > 10 and elapsed_time <= 30:  # Exit if running for more than 10 seconds
                print("Timeout: Exiting the loop after 10 seconds.")

                try:
                    print('into try loop')
                    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + 'freegame')

                    # try template
                    freegame_screenshot = os.path.join(root_dir, 'images', Snapshot + 'freegame.png')
                    all_freegame_btn_json_path = os.path.join(root_dir, 'marquee_tool', Snapshot + '_runtime',
                                                              Snapshot + '_runtime_regions.json')
                    all_freegame_btn_json = json.load(open(all_freegame_btn_json_path, encoding='utf=8'))
                    matched_loc = test_template_matching(freegame_screenshot, all_freegame_btn_json)
                    if len(matched_loc) > 0:
                        loc = random.choice(matched_loc)
                        GameController.click_in_window(window_title=window_name, x_offset=loc[0] + loc[2] // 2,
                                                       y_offset=loc[1] + loc[3] // 2)
                        break

                    # Segment the image
                    maskDict_path = os.path.join(images_dir, Snapshot + 'freegame' + ".png")
                    maskDict = sam.segment_image(maskDict_path)

                    # Classify components
                    vit = ViTRecognition(
                        Snapshot=Snapshot,
                        maskDict=maskDict,
                        model_path=vit_model_path
                    )
                    highest_confidence_images, template_folder = vit.classify_components()
                    vit.output_json(template_folder=template_folder,
                                    highest_confidence_images=highest_confidence_images)

                    # Check for specific predictions
                    if any(key in [3, 8, 12, 13] for key in highest_confidence_images.keys()):
                        key = [key in [3, 8, 12, 13] for key in highest_confidence_images.keys()]
                        print('detecting 金元寶或金幣')
                        GameController.Windowcontrol(GameController,
                                                     highest_confidence_images=highest_confidence_images,
                                                     classId=key[0])
                        break  # Exit the loop

                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue  # Skip to the next iteration if an error occurs

                # Capture screenshot with timestamp in filename
                timestamp = datetime.now().strftime('%m%d_%H%M%S')  # Format: MMDD_HHMMSS
                filename = Snapshot + '_runtime_' + timestamp + '.png'
                screenshot.capture_screenshot(window_title=window_name, filename=filename)
                print(f"Screenshot saved as {filename}")
                break
            elif elapsed_time > 30:
                break


if __name__ == "__main__":
    main()
