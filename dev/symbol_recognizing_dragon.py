import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
from Symbol_recognition.grid_recognizer import BaseGridRecognizer

MODE = 'base'
GAME = 'dragon'
DEBUG = False

image_dir = Path(f"./images/{GAME}/screenshots/{MODE}_game")
config_file = Path(f'./Symbol_recognition/configs/{GAME}.json')
grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=(1920, 1080), debug=DEBUG)


frame_count = 0 # replace it when integrating with the game
for image_path in image_dir.glob('*.png'): 
    image_name = image_path.stem
    if DEBUG and image_name != "6":
        continue
    print(f"Processing image: {image_name}")
    img = cv2.imread(str(image_path))
    
    grid_recognizer.initialize_grid(img)
    grid_recognizer.recognize_roi(img, 2)
    grid_recognizer.save_annotated_frame(img, image_name)
    grid_recognizer.save_grid_results(str(frame_count))

    frame_count += 1
    print("------------------------------------------------------")