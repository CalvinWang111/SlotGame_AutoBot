from value_recognition import ValueRecognition
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from PIL import Image
import time
import value_recognition

def main():
    # 初始化模組
    screenshot = GameScreenshot()
    value_recognition = ValueRecognition()
    window_name = 'BlueStacks App Player'
    Snapshot = 'inputTest'
    intensity_threshold = 20
    spin_round = 20

    sam = SAMSegmentation(Snapshot=Snapshot)


    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)

    # 2. SAM 分割
    maskDict = sam.segment_image(r"./images/" + Snapshot + ".png")

    # 3. ViT 辨識
    # put your own VIT model path here
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict,
                         model_path=r'D:\git-repository\SlotGame_AutoBot\SlotBot\VITrun_ver6\best_model.pth')
    highest_confidence_images, template_folder = vit.classify_components()

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + '_runtime')
    intial_intensity = screenshot.clickable(snapshot_path=r"./images/" + Snapshot + "_runtime.png",
                                            highest_confidence_images=highest_confidence_images)
    intensity = intial_intensity + intensity_threshold

    for i in range(spin_round):
        time.sleep(1)
        screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + '_runtime')
        value_recognition.get_board_value("./images/" + Snapshot + "_runtime.png")
        GameController.Windowcontrol(GameController, highest_confidence_images=highest_confidence_images, classId=8)
        print('spin')
        time.sleep(3)

        if i == 10:
            value_recognition.get_meaning()

        while (abs(intial_intensity - intensity) >= intensity_threshold):
            screenshot.capture_screenshot(window_title=window_name, filename=Snapshot + '_runtime')
            intensity = screenshot.clickable(snapshot_path=r"./images/" + Snapshot + "_runtime.png",
                                             highest_confidence_images=highest_confidence_images)
            print('waiting')



if __name__ == "__main__":
    main()
