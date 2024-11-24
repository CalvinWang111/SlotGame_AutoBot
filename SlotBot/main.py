from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from symbol_recognizing import *
from stopping_detection import StoppingFrameCapture

from PIL import Image
import time
from grid import BullGrid
import cv2
from queue import Queue
import threading
from pathlib import Path


MODE = 'base'
GAME = 'dragon'

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
    Snapshot = 'inputTest'
    intensity_threshold = 20
    spin_round = 20
    cell_border = 20

    sam = SAMSegmentation(Snapshot=Snapshot)
    
    
    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(r"./images/"+Snapshot+".png")
    print("segment completed")

    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=r'../best_model.pth')
    highest_confidence_images, template_folder = vit.classify_components()
    print("ViTRecognition completed")
    
    # 4. 根據第一張畫面初始化程式
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_runtime')
    intial_intensity = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_runtime.png",highest_confidence_images=highest_confidence_images)
    
    first_frame = cv2.imread(r"./images/"+Snapshot+"_runtime.png")

    matched_positions = get_symbol_positions(template_dir=symbol_template_dir, image=first_frame)
    if matched_positions is None:
        return -1
    grid_bbox, grid_shape = get_grid_info(matched_positions)
    grid = BullGrid(grid_bbox, grid_shape, MODE)
    print(f'initial grid shape: {grid.row} x {grid.col}')

    stop_catcher = StoppingFrameCapture(window_name=window_name,grid=grid,save_dir=key_frame_dir)

    output_counter = 0
    for round_number in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=8)
        key_frame_pathes = stop_catcher.get_key_frames(intial_intensity,intensity_threshold,highest_confidence_images)
        
        # process key frames
        for path in key_frame_pathes:
            img = cv2.imread(path)
            grid = recoglize_symbol(img=img,grid=grid,template_dir=symbol_template_dir,game_mode=MODE,cell_border=cell_border)
            save_path = save_dir / f"capture_result{output_counter}.png"
            output_counter += 1
            draw_bboxes_and_icons_on_image(img, symbol_template_dir, grid, save_path=save_path)
            grid.clear()
            
if __name__ == "__main__":
    main()
