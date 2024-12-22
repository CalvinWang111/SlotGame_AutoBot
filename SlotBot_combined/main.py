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
from value_recognition import ValueRecognition
import threading
import cProfile
import pstats

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
    intensity_threshold = 20
    cell_border = 20
    spin_round = 100
    value_recognize_signal = False
    root_dir = Path(__file__).parent.parent
    print('rootdir',root_dir)

    vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
    sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
    #sam_model_cfg = os.path.join(root_dir, 'sam2','sam2_configs', 'sam2_hiera_l.yaml')
    images_dir = os.path.join(root_dir, 'images')
    
    print('sam_model_path',sam_model_path)
    print('sam_model_cfg',sam_model_cfg)

    sam = SAMSegmentation(Snapshot=Snapshot, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)
    valuerec = ValueRecognition()

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
    valuerec.get_board_value(intialshot_path)

    config_file = Path(root_dir / f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
    grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=(1920, 1080), debug=False)
    grid_recognizer.initialize_grid(first_frame)
    # temp_img = draw_grid_on_image(first_frame, grid_recognizer.grid)
    # cv2.imshow('grid', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    stop_catcher = StoppingFrameCapture(window_name=window_name,grid=grid_recognizer.grid,save_dir=key_frame_dir, Snapshot=Snapshot, elapsed_time_threshold=3, game_name=GAME)
    numerical_round_count = 0

    def keyframes_wrapper(module_instance, key_frame_pathes):
        key_frame_pathes = stop_catcher.get_key_frames(intial_intensity=intial_avg_intensities,intensity_threshold=intensity_threshold,highest_confidence_images=highest_confidence_images)
    def profiled_keyframes_wrapper(module_instance, key_frame_pathes):
        profiler = cProfile.Profile()
        profiler.enable()
        keyframes_wrapper(module_instance, key_frame_pathes)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')  # 按累計時間排序
        stats.print_stats(10)  

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
    
    
        '''
        key_frame_pathes = stop_catcher.get_key_frames(intial_intensity=intial_avg_intensities,intensity_threshold=intensity_threshold,highest_confidence_images=highest_confidence_images)

        '''
        key_frame_pathes = []
        stop_catcher_thread = threading.Thread(target=profiled_keyframes_wrapper, args=(stop_catcher, key_frame_pathes))
        stop_catcher_thread.start()

        while stop_catcher_thread.is_alive():
            time.sleep(1)
            if stop_catcher.free_gamestate:
                print('超過10秒未能恢復操作，判定已經進入免費遊戲')
        
        # process key frames
        for path in key_frame_pathes:
            key_frame_name = Path(path).stem
            print(f'Processing key frame: {key_frame_name}')
            img = cv2.imread(path)
            grid_recognizer.initialize_grid(img)
            grid_recognizer.recognize_roi(img, 2)
            grid_recognizer.save_annotated_frame(img, key_frame_name)
            grid_recognizer.save_grid_results(key_frame_name)
            
            #cv2.imshow('key_frame', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            numerical_round_count += 1
            if value_recognize_signal:
                valuerec.recognize_value([path])

            # save_path = save_dir / f"capture_result{output_counter}.png"
            # output_counter += 1
            # symbol_recognizer.draw_bboxes_and_icons_on_image(img, symbol_template_dir, grid, save_path=save_path)
            # grid.clear()
            
        print(key_frame_dir, numerical_round_count)
        #數值組 
        if i == 10:
            all_keyframes = [os.path.join(key_frame_dir, file) for file in os.listdir(key_frame_dir)]
            # Sort the files if needed (e.g., alphabetically or by modification time)
            all_keyframes.sort()

             # Collect the first `file_count` files
            all_keyframes = all_keyframes[:numerical_round_count]
            # numerical_round_cound減少1，key_frame編號記錄從0開始，round從1開始
            valuerec.get_meaning(all_keyframes, numerical_round_count - 1)
            value_recognize_signal = True

if __name__ == "__main__":
    main()
