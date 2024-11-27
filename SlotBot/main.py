import os
import time
from pathlib import Path
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from TemplatMatching.template_matching import test_template_matching
from PIL import Image



def main():
    # 初始化模組
    root_dir = Path(__file__).parent.parent
    vit_model = os.path.join(root_dir, 'VITrun_ver6', 'best_model.pth')
    sam_model = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    # sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')

    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'inputTest'
    images_dir = os.path.join(root_dir, 'images')
    intensity_threshold = 20
    spin_round = 20
    is_region_selected = False

    sam = SAMSegmentation(Snapshot=Snapshot, sam2_checkpoint=sam_model)

    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)

    # 2. SAM 分割
    sam_input_image = os.path.join(images_dir, Snapshot + '.png')

    maskDict = sam.segment_image(sam_input_image)
    
    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=vit_model)
    highest_confidence_images, template_folder = vit.classify_components()

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_runtime')
    intial_intensity = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_runtime.png",highest_confidence_images=highest_confidence_images)
    intensity = intial_intensity + intensity_threshold

    for i in range(spin_round):
        
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=8)

        print('spin')
        time.sleep(3)

        while(abs(intial_intensity - intensity) >= intensity_threshold):
            screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_runtime')
            intensity = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_runtime.png",highest_confidence_images=highest_confidence_images)
            print('waiting')

            if not is_region_selected:
                # selected button region
                is_region_selected = True
                
            else:
                # call selection tool

                # all_template_path = None
                # test_template_matching(os.path.join(root_dir, 'images', Snapshot + '_runtime.png'), all_template_path)
                
                pass

if __name__ == "__main__":
    main()
