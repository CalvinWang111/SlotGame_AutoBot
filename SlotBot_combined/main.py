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
    Snapshot = 'FuXin'
    intensity_threshold = 10
    cell_border = 20
    spin_round = 3
    root_dir = Path(__file__).parent.parent

    vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
    sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
    images_dir = os.path.join(root_dir, 'images')


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

    intial_avg_intensities = screenshot.clickable(snapshot_path=intialshot_path,highest_confidence_images=highest_confidence_images)
    first_frame = cv2.imread(intialshot_path)

    matched_positions = get_symbol_positions(template_dir=symbol_template_dir, image=first_frame)

    if matched_positions is None:
        return -1
    grid_bbox, grid_shape = symbol_recognizer.get_grid_info(matched_positions)
    grid = grid.BullGrid(grid_bbox, grid_shape, MODE)

    stop_catcher = StoppingFrameCapture(window_name=window_name,grid=grid,save_dir=key_frame_dir, Snapshot=Snapshot)

    for i in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=10)
        print('spin round : ',i)
        time.sleep(3)

        #設定初始值，以此進入while迴圈
        avg_intensities = {
            class_id: [value + intensity_threshold for value in intensities]
            for class_id, intensities in intial_avg_intensities.items()
        }

        key_frame_pathes = stop_catcher.get_key_frames(intial_intensity=avg_intensities,intensity_threshold=intensity_threshold,highest_confidence_images=highest_confidence_images)
        print(stop_catcher.state)

        # process key frames
        for path in key_frame_pathes:
            img = cv2.imread(path)
            grid = recoglize_symbol(img=img,grid=grid,template_dir=symbol_template_dir,game_mode=MODE,cell_border=cell_border)
            save_path = save_dir / f"capture_result{output_counter}.png"
            output_counter += 1
            symbol_recognizer.draw_bboxes_and_icons_on_image(img, symbol_template_dir, grid, save_path=save_path)
            grid.clear()

if __name__ == "__main__":
    main()
