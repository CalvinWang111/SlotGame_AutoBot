from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from PIL import Image

def main():
    # 初始化模組
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'inputTest'

    sam = SAMSegmentation(Snapshot=Snapshot)
    
    
    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(r"./images/"+Snapshot+".png")
    
    # 3. ViT 辨識
    # put your own VIT model path here 
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=r'C:\Users\13514\button_recognition\VITrun_ver6\best_model.pth')
    highest_confidence_images = vit.classify_components()

    # 4. 操控遊戲
    GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=8)

if __name__ == "__main__":
    main()
