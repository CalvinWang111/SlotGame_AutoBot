import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
from SlotBot_combined.Symbol_recognition.grid_recognizer import BaseGridRecognizer

MODE = 'base'
GAME = 'Plusz_LegendOfRa'
DEBUG = False

vertical_size = (1080, 1920)
horizontal_size = (1920, 1080)

image_dir = Path(f"./images/{GAME}/screenshots/{MODE}_game")
config_file = Path(f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=horizontal_size, debug=DEBUG)


frame_count = 0 # replace it when integrating with the game
for image_path in image_dir.glob('*.png'): 
    image_name = image_path.stem
    # if DEBUG and image_name != "6":
    #     continue
    print(f"Processing image: {image_name}")
    print(f'image size: {cv2.imread(str(image_path)).shape}')
    img = cv2.imread(str(image_path))
    
    grid_recognizer.initialize_grid(img)
    grid_recognizer.recognize_roi(img, 1) # 0 for template matching, 1 for SIFT matching
    grid_recognizer.save_annotated_frame(img, image_name)
    grid_recognizer.save_grid_results(str(frame_count))

    frame_count += 1
    print("------------------------------------------------------")