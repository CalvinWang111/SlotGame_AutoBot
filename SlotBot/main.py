from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from stopping_detection import StoppingFrameCapture
from PIL import Image
import time
from grid import BullGrid
import cv2
from queue import Queue
import threading

def main():
    # 初始化模組
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'inputTest'
    intensity_threshold = 20
    spin_round = 10

    sam = SAMSegmentation(Snapshot=Snapshot)
    
    
    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(r"./images/"+Snapshot+".png")
    
    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=r'../best_model.pth')
    highest_confidence_images, template_folder = vit.classify_components()

    print(highest_confidence_images)
    print(highest_confidence_images.items())
    # cv2.imshow("highest_confidence_images",highest_confidence_images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_runtime')
    intial_intensity = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_runtime.png",highest_confidence_images=highest_confidence_images)
    
    intensity = intial_intensity + intensity_threshold
    frame = cv2.imread(r"./images/"+Snapshot+"_runtime.png")
    
    # player_display_roi = cv2.selectROIs('Select player display', frame, showCrosshair=False, fromCenter=False)[0]
    player_display_roi = [257, 181, 694, 374]
    print("roi: ",player_display_roi)
    cv2.destroyAllWindows()
    grid = BullGrid(player_display_roi,(4,5))
    stop_catcher = StoppingFrameCapture(window_name,"./key_frames",grid)

    
    for i in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=8)
        print('spin')
        stop_catcher.save_key_frames(intial_intensity,highest_confidence_images,intensity_threshold)
        # time.sleep(3)
        # intensity = intial_intensity + intensity_threshold
        # while(1):
        #     latest_frame = stop_catcher.get_latest_frame()
        #     print(latest_frame)
        #     if latest_frame is not None:
        #         intensity = screenshot.clickable(latest_frame,highest_confidence_images=highest_confidence_images)
        #         if abs(intial_intensity - intensity) < intensity_threshold:
        #             stop = [True]
        #             break
        #     time.sleep(0.01)
            
if __name__ == "__main__":
    main()
