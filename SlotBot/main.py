from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from component_matching import ComponentMatcher
from game_controller import GameController
from PIL import Image
import time
from datetime import datetime

def main():
    # 初始化模組
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = 'FuXin'
    intensity_threshold = 10
    spin_round = 500

    sam = SAMSegmentation(Snapshot=Snapshot)
    
    
    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(r"./images/"+Snapshot+".png")
    
    # 3. ViT 辨識
    # put your own VIT model path here 
    #vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict, model_path=r'C:\Users\13514\button_recognition\VITrun_ver6\best_model.pth')
    vit = ViTRecognition(Snapshot=Snapshot, maskDict=maskDict,model_path=r"C:\Users\13514\button_recognition\VITrun_ver7\vit_model.pth")
    highest_confidence_images, template_folder = vit.classify_components()

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_intialshot')
    intial_avg_intensities = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_intialshot.png",highest_confidence_images=highest_confidence_images)

    for i in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=10)
        print('spin round : ',i)
        time.sleep(1)
        start_time = time.time()

        #設定初始值，以此進入while迴圈
        avg_intensities = {
            class_id: [value + intensity_threshold for value in intensities]
            for class_id, intensities in intial_avg_intensities.items()
        }

        while(screenshot.intensity_check(intial_avg_intensities,avg_intensities,intensity_threshold)):
            
            elapsed_time = time.time() - start_time
            print('elapsed time', elapsed_time)
            screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'_runtime')
            avg_intensities = screenshot.clickable(snapshot_path=r"./images/"+Snapshot+"_runtime.png",highest_confidence_images=highest_confidence_images)
            print('waiting')

            if elapsed_time > 10:  # Exit if running for more than 10 seconds
                print("Timeout: Exiting the loop after 10 seconds.")
                
                try:
                    print('into try loop')
                    screenshot.capture_screenshot(window_title=window_name, filename=Snapshot+'freegame')

                    # Segment the image
                    maskDict = sam.segment_image(r"./images/" + Snapshot+'freegame' + ".png")
                    
                    # Classify components
                    vit = ViTRecognition(
                        Snapshot=Snapshot, 
                        maskDict=maskDict, 
                        model_path=r"C:\Users\13514\button_recognition\VITrun_ver7\vit_model.pth"
                    )
                    highest_confidence_images, template_folder = vit.classify_components()
                    
                    
                    # Check for specific predictions
                    if any(key in [12, 13] for key in highest_confidence_images.keys()):
                        key = [key in [12,13] for key in highest_confidence_images.keys()]
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
            


if __name__ == "__main__":
    main()
