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
from TemplateMatching.template_matching import test_template_matching
from PIL import Image
from stopping_detection import StoppingFrameCapture
from TemplateMatching import symbol_recognizer,grid
from symbol_recognizing import get_symbol_positions,recoglize_symbol
from Symbol_recognition.grid_recognizer import *
import cv2

MODE = 'base'
GAME = 'golden'

if MODE == 'base':
    symbol_template_dir = Path(f'./images/{GAME}/symbols/base_game')
    image_dir = Path(f'./images/{GAME}/screenshots/base_game')
    save_dir = Path(f'./temp/{GAME}_base_output')
elif MODE == 'free':
    symbol_template_dir = Path(f'./images/{GAME}/symbols/free_game')
    image_dir = Path(f'./images/{GAME}/screenshots/free_game')
    save_dir = Path(f'./temp/{GAME}_free_output')
key_frame_dir = Path(f'./temp/key_frame')
save_dir.mkdir(parents=True, exist_ok=True)
key_frame_dir.mkdir(parents=True, exist_ok=True)


def main():
    # 初始化模組
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'GoldenHoYeah'
    intensity_threshold = 10
    cell_border = 20
    spin_round = 3
    root_dir = Path(__file__).parent.parent
    print('rootdir',root_dir)

    vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
    sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    #sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
    sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
    images_dir = os.path.join(root_dir, 'images')
    
    print('sam_model_path',sam_model_path)
    print('sam_model_cfg',sam_model_cfg)

    sam = SAMSegmentation(Snapshot=Snapshot, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)

    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(os.path.join(root_dir, 'images', Snapshot + ".png"))
    
    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict,model_path=vit_model_path)
    highest_confidence_images, template_folder = vit.classify_components()
    vit.output_json(template_folder=template_folder, highest_confidence_images=highest_confidence_images)


    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_intialshot')

    intialshot_path = os.path.join(images_dir, Snapshot+"_intialshot.png")
    print('intialshot_path', intialshot_path)
    intial_avg_intensities = screenshot.clickable(snapshot_path=intialshot_path,highest_confidence_images=highest_confidence_images)
    first_frame = cv2.imread(intialshot_path)
    #first_frame = cv2.imread(r"./images/"+Snapshot+"_runtime.png")
    print(first_frame, type(first_frame))

    config_file = Path(root_dir / f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
    grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=(1920, 1080), debug=False)
    grid_recognizer.initialize_grid(first_frame)
    # temp_img = draw_grid_on_image(first_frame, grid_recognizer.grid)
    # cv2.imshow('grid', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    stop_catcher = StoppingFrameCapture(window_name=window_name,grid=grid_recognizer.grid,save_dir=key_frame_dir, Snapshot=Snapshot, elapsed_time_threshold=3)

    for i in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=10)
        print('spin round : ',i)
        #time.sleep(3)
        start_time = time.time()

        #設定初始值，以此進入while迴圈
        avg_intensities = {
            class_id: [value + intensity_threshold for value in intensities]
            for class_id, intensities in intial_avg_intensities.items()
        }

        key_frame_pathes = stop_catcher.get_key_frames(intial_intensity=intial_avg_intensities,intensity_threshold=intensity_threshold,highest_confidence_images=highest_confidence_images)
        
        # process key frames
        for path in key_frame_pathes:
            key_frame_name = Path(path).stem
            print(f'Processing key frame: {key_frame_name}')
            img = cv2.imread(path)
            # grid_recognizer.initialize_grid(img)
            # grid_recognizer.recognize_roi(img, 2)
            # grid_recognizer.save_annotated_frame(img, key_frame_name)
            # grid_recognizer.save_grid_results(key_frame_name)
            cv2.imshow('key_frame', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # save_path = save_dir / f"capture_result{output_counter}.png"
            # output_counter += 1
            # symbol_recognizer.draw_bboxes_and_icons_on_image(img, symbol_template_dir, grid, save_path=save_path)
            # grid.clear()
        '''
        while(stop_catcher.get_key_frames(intial_intensity=avg_intensities, intensity_threshold=intensity_threshold, highest_confidence_images=highest_confidence_images)):
        #while(screenshot.intensity_check(intial_avg_intensities,avg_intensities,intensity_threshold)):

            elapsed_time = time.time() - start_time
            print('elapsed time', elapsed_time)


            key_frame_pathes = stop_catcher.get_key_frames(intial_intensity,intensity_threshold,highest_confidence_images)
        
            # process key frames
            for path in key_frame_pathes:
                img = cv2.imread(path)
                grid = recoglize_symbol(img=img,grid=grid,template_dir=symbol_template_dir,game_mode=MODE,cell_border=cell_border)
                save_path = save_dir / f"capture_result{output_counter}.png"
                output_counter += 1
                draw_bboxes_and_icons_on_image(img, symbol_template_dir, grid, save_path=save_path)
                grid.clear()


            if elapsed_time > 10 and elapsed_time <= 30:  # Exit if running for more than 10 seconds
                print("Timeout: Exiting the loop after 10 seconds.")

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
                        break

                    # Segment the image
                    maskDict_path = os.path.join(images_dir, Snapshot+'freegame' + ".png")
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
        '''

if __name__ == "__main__":
    main()
