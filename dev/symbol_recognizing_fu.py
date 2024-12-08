import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
import time
from Symbol_recognition.grid import BaseGrid
from Symbol_recognition.symbol_recognizer import *
from Symbol_recognition.utils import *

MODE = 'base'
GAME = 'fu'

output_json_dir = Path(f'./output/{GAME}/{MODE}')
if not output_json_dir.exists():
    output_json_dir.mkdir(parents=True, exist_ok=True)
    
if MODE == 'base':
    template_dir = Path(f'./images/{GAME}/symbols/base_game')
    image_dir = Path(f'./images/{GAME}/screenshots/base_game')
    save_dir = Path(f'./temp/{GAME}_base_output')
elif MODE == 'free':
    template_dir = Path(f'./images/{GAME}/symbols/free_game')
    image_dir = Path(f'./images/{GAME}/screenshots/free_game')
    save_dir = Path(f'./temp/{GAME}_free_output')
save_dir.mkdir(parents=True, exist_ok=True)

grid = None
grid_path = Path(f'./dev/grid/{GAME}_{MODE}_grid.pkl')
if grid_path.exists():
    grid = BaseGrid.load(str(grid_path))

template_match_data = {}
cell_border = 10

frame_count = 0 # replace it when integrating with the game
for image_path in image_dir.glob('*.png'): 
    print(f"Processing image: {image_path}")
    img = cv2.imread(str(image_path))
    
    # initialize grid
    if grid is None:
        start_time = time.time()
        process_template_matches(
            template_match_data=template_match_data, 
            template_dir=template_dir, 
            img=img, 
            iou_threshold=0.1, 
            scale_range=[0.9, 1.3],
            scale_step=0.05,
            threshold=0.95,
            min_area=5000,
            border=100
        )
        elapsed_time = time.time() - start_time
        print(f"Initial grid matching: {elapsed_time:.2f} seconds")
        
        matched_positions = []
        for template_name, data in template_match_data.items():
            w, h = data['shape'][1], data['shape'][0]
            for (top_left, scale, _) in data['result']:
                x = top_left[0] + w * scale / 2
                y = top_left[1] + h * scale / 2
                matched_positions.append((x, y))
        if len(matched_positions) == 0:
            print("Could not find any matches")
            break
        print(f'found {len(matched_positions)} symbols')
                
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        grid = BaseGrid(grid_bbox, grid_shape)
        grid.save(str(grid_path))
        print(f'initial grid shape: {grid.row} x {grid.col}')
    
    # Process each grid cell
    start_time = time.time()
    for j in range(grid.col):        
        for i in range(grid.row):
            roi = grid.get_roi(i, j)
            x, y, w, h = roi
            x1, x2 = x - cell_border, x + w + cell_border
            y1, y2 = y - cell_border, y + h + cell_border
            cell = img[y1:y2, x1:x2]
            
            symbol_name, score = process_template_matches(
                template_match_data=template_match_data,
                template_dir=template_dir,
                img=cell,
                iou_threshold=0.1,
                scale_range=[0.8, 1.5],
                scale_step=0.05,
                threshold=0.8,
                min_area=5000,
                match_one=True,
                border=cell_border,
            )
            grid[i, j] = {"symbol": symbol_name, "score": score, "value": None}
            
    elapsed_time = time.time() - start_time
    print(f"Grid cells matching: {elapsed_time:.2f} seconds")
    
    save_path = save_dir / f"{image_path.stem}.png"
    draw_bboxes_and_icons_on_image(img, template_dir, grid, save_path=save_path)
    
    # save output json
    grid.save_results_as_json(save_dir=output_json_dir, template_dir=template_dir, frame_count=frame_count)
    
    # clear grid
    grid.clear()
    frame_count += 1
    print("------------------------------------------------------")