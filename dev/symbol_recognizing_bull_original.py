import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
import time
from Symbol_recognition.grid import BullGrid
from Symbol_recognition.symbol_recognizer import *
from Symbol_recognition.utils import *

MODE = 'base'
GAME = 'bull'
direction = 'up' if MODE == 'base' else 'down'

if MODE == 'base':
    template_dir = Path(f'./images/{GAME}/symbols/base_game')
    image_dir = Path(f'./images/{GAME}/screenshots/base_game')
elif MODE == 'free':
    template_dir = Path(f'./images/{GAME}/symbols/free_game')
    image_dir = Path(f'./images/{GAME}/screenshots/free_game')

template_match_data = {}
grid = None
is_init_column_heights = False
cell_border = 10

if MODE == 'base':
    arrow_name = "up_arrow"
elif MODE == 'free':
    arrow_name = "down_arrow"
arrow_template = cv2.imread(str(template_dir / f'{arrow_name}.png'), cv2.IMREAD_UNCHANGED)
arrow_scale_range = [1.0, 2.0]


for image_path in image_dir.glob('*.png'): 
    print(f"Processing image: {image_path}")
    img = cv2.imread(str(image_path))
    
    # initialize grid
    if grid is None:
        start_time = time.time()
        process_template_matches(
            template_match_data=template_match_data, 
            template_dir=template_dir, 
            roi=img, 
            iou_threshold=0.1, 
            scale_range=[0.9, 1.5],
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
        if len(matched_positions) == 0:
            print("Could not find any matches")
            break
        print(f'found {len(matched_positions)} symbols')
                
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        grid = BullGrid(grid_bbox, grid_shape, direction)
        print(f'initial grid shape: {grid.row} x {grid.col}')

        draw_grid_on_image(img, grid)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        cv2.imshow('grid', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Process each grid cell
    start_time = time.time()
    for j in range(grid.col):
        row_range = (
            range(grid.row - grid.column_heights[j], grid.row) if MODE == 'base'
            else range(grid.column_heights[j])
        )
        
        for i in row_range:
            roi = grid.get_roi(i, j)
            x, y, w, h = roi
            x1, x2 = x - cell_border, x + w + cell_border
            y1, y2 = y - cell_border, y + h + cell_border
            cell = img[y1:y2, x1:x2]
            
            symbol_name = process_template_matches(
                template_match_data=template_match_data,
                template_dir=template_dir,
                roi=cell,
                iou_threshold=0.1,
                scale_range=[0.9, 1.5],
                scale_step=0.02,
                threshold=0.88,
                min_area=5000,
                match_one=True,
                border=cell_border,
            )
            
            grid[i, j] = symbol_name
    
    if not is_init_column_heights:
        grid.init_column_heights()
        is_init_column_heights = True
        print("initial column heights:", grid.column_heights)
        
    elapsed_time = time.time() - start_time
    print(f"Grid cells matching: {elapsed_time:.2f} seconds")
    
    #find arrow at each column
    start_time = time.time()                    
    for j in range(grid.col):
        if MODE == 'base':
            index = grid.row - grid.column_heights[j] - 1
            position = 'free'
        elif MODE == 'free':
            index = grid.column_heights[j]
            position = 'bottom'

        roi = grid.get_roi(index, j)
        x, y, w, h = roi
        match_result = template_matching(arrow_template, img[y:y+h, x:x+w], scale_range=arrow_scale_range, scale_step=0.05, threshold=0.90, border=0)
        match_result, best_scale = apply_nms_and_filter_by_best_scale(match_result, arrow_template.shape, iou_threshold=0.1)

        if len(match_result) > 0:
            arrow_scale_range = [best_scale, best_scale]
            grid.column_heights[j] += 1
            if grid.column_heights[j] > grid.max_height:
                grid.column_heights[j] = grid.base_height
            print(f'{direction} arrow found at column[{j}], update column_heights[{j}] to {grid.column_heights[j]}')

            if grid.column_heights[j] > grid.row:
                grid.add_row(position=position)
                print(f'Updated grid shape: {grid.row} x {grid.col}')
                
    for j in range(grid.col):
        print(f'column[{j}] height: {grid.column_heights[j]}')
        
    elapsed_time = time.time() - start_time
    print(f"Up arrow matching: {elapsed_time:.2f} seconds")
    
    draw_bboxes_and_icons_on_image(img, template_dir, grid, save_path=f"temp/{image_path.stem}.png")
    
    grid.clear()
    
    print("------------------------------------------------------")