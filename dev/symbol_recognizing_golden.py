import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
from SlotBot_combined.Symbol_recognition.grid_recognizer import BaseGridRecognizer

MODE = 'base'
GAME = 'golden'
DEBUG = False

image_dir = Path(f"./images/{GAME}/screenshots/{MODE}_game")
image_dir = Path(f"./test_images/")
config_file = Path(f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
grid_recognizer = None



frame_count = 0 # replace it when integrating with the game
for image_path in image_dir.glob('*.png'): 
    image_name = image_path.stem
    # if DEBUG and image_name != "2":
    #     continue
    print(f"Processing image: {image_name}")
    img = cv2.imread(str(image_path))
    print(f'image size: {img.shape}')
    
    if grid_recognizer is None:
        grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=(img.shape[1], img.shape[0]), debug=DEBUG)
    
    grid_recognizer.initialize_grid(img)
    grid_recognizer.recognize_roi(img, 0)
    grid_recognizer.save_annotated_frame(img, image_name)
    grid_recognizer.save_grid_results(str(frame_count))

    frame_count += 1
    print("------------------------------------------------------")