import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from Symbol_recognition.grid import BaseGrid
from Symbol_recognition.utils import *
from Symbol_recognition.symbol_recognizer import *

GAME = 'golden'
MODE = 'free'
DEBUG = False

grid = None
border = 0
template_match_data = {}

template_dir = Path(f'images/{GAME}/symbols/free_game')
img_dir = Path(f'images/{GAME}/screenshots/free_game')
output_dir = Path(f'temp/{GAME}_free_game_output')
output_dir.mkdir(parents=True, exist_ok=True)
grid_path = Path(f'dev/grid/{GAME}_{MODE}_grid.pkl')

if grid_path.exists():
    grid = BaseGrid.load(str(grid_path))

for img_path in img_dir.iterdir():
    if DEBUG:
        break
    img_name = img_path.stem
    img = cv2.imread(str(img_path))
    print(f'Processing {img_name}...')
    
    # initialize grid
    if grid is None:
        process_template_matches(
            template_match_data=template_match_data, 
            template_dir=template_dir, 
            img=img, 
            iou_threshold=0.1, 
            scale_range=[0.7, 1.5],
            scale_step=0.05,
            threshold=0.94,
            min_area=3000,
            border=border,
            grayscale=True
        )
        
        matched_positions = []
        for template_name, data in template_match_data.items():
            w, h = data['shape'][1], data['shape'][0]
            for (top_left, scale, score) in data['result']:
                x = top_left[0] + w * scale / 2
                y = top_left[1] + h * scale / 2
                matched_positions.append((x, y))
                # roi = img[top_left[1]:top_left[1]+int(h*scale), top_left[0]:top_left[0]+int(w*scale)]
                # print(f'template_name: {template_name}, scale: {scale}, area: {w*h*scale*scale}, score: {score}')
                # cv2.imshow('roi', roi)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        if len(matched_positions) == 0:
            print("Could not find any matches")
            break
        print(f'Found {len(matched_positions)} matches')
                
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        grid = BaseGrid(grid_bbox, grid_shape)
        grid.save(str(grid_path))
        print(f'initial grid shape: {grid.row} x {grid.col}')
        
        # draw_grid_on_image(img, grid)
        # img_copy = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        # cv2.imshow('grid', img_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for i in range(grid.row):
        for j in range(grid.col):
            x, y, w, h = grid.get_roi(i, j)
            roi = img[y-border:y+h+border, x-border:x+w+border]

            best_match, best_score, num_matches = process_template_matches_sift(template_dir, roi, (0.5, 1.5), False)
            grid[i, j] = best_match

    output_path = output_dir / f'{img_name}.png'
    draw_bboxes_and_icons_on_image(img, template_dir, grid, str(output_path))

img_path = Path(f'images/{GAME}/screenshots/free_game/Screenshot_2024.12.02_10.26.03.780.png')
img_name = img_path.stem
img = cv2.imread(str(img_path))

if DEBUG:
    for i in range(grid.row):
        for j in range(grid.col):

            x, y, w, h = grid.get_roi(i, j)
            roi = img[y-border:y+h+border, x-border:x+w+border]
            # roi = img[y:y+h, x:x+w] 
            
            best_match, best_scale, num_matches = process_template_matches_sift(template_dir, roi, (0.5, 1.5), True)
            grid[i, j] = best_match
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()