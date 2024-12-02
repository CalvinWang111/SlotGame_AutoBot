import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
import time
from TemplateMatching.grid import BaseGrid
from TemplateMatching.symbol_recognizer import *
from TemplateMatching.utils import *

MODE = 'base'
GAME = 'golden'

if MODE == 'base':
    template_dir = Path(f'./images/{GAME}/symbols/base_game')
    image_dir = Path(f'./images/{GAME}/screenshots/base_game')
    save_dir = Path(f'./temp/{GAME}_base_output')
elif MODE == 'free':
    template_dir = Path(f'./images/{GAME}/symbols/free_game')
    image_dir = Path(f'./images/{GAME}/screenshots/free_game')
    save_dir = Path(f'./temp/{GAME}_free_output')
save_dir.mkdir(parents=True, exist_ok=True)

template_match_data = {}
grid = None
cell_border = 10

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
            scale_range=[0.5, 1.5],
            scale_step=0.05,
            threshold=0.95,
            min_area=3000,
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
                print(f'{template_name}: area: {w * h * scale * scale})')
        if len(matched_positions) == 0:
            print("Could not find any matches")
            break
        print(f'found {len(matched_positions)} symbols')
                
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        grid = BaseGrid(grid_bbox, grid_shape)
        print(f'initial grid shape: {grid.row} x {grid.col}')
        
        # draw_grid_on_image(img, grid)
        # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        # cv2.imshow('grid', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    
    # Process each grid cell
    start_time = time.time()
    for j in range(grid.col):
        for i in range(grid.row):
            roi = grid.get_roi(i, j)
            x, y, w, h = roi
            x1, x2 = x - cell_border, x + w + cell_border
            y1, y2 = y - cell_border, y + h + cell_border
            cell = img[y1:y2, x1:x2]
            
            symbol_name = process_template_matches(
                template_match_data=template_match_data,
                template_dir=template_dir,
                img=cell,
                iou_threshold=0.1,
                scale_range=[0.5, 1.5],
                scale_step=0.05,
                threshold=0.8,
                min_area=3000,
                match_one=True,
                border=cell_border,
            )
            
            grid[i, j] = symbol_name
    elapsed_time = time.time() - start_time
    print(f"Grid cells matching: {elapsed_time:.2f} seconds")
    
    save_path = save_dir / f"{image_path.stem}.png"
    draw_bboxes_and_icons_on_image(img, template_dir, grid, save_path=save_path)
    grid.clear()
    print("------------------------------------------------------")