import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from TemplateMatching.grid import BaseGrid
from TemplateMatching.utils import *
from TemplateMatching.symbol_recognizer import *

GAME = 'dragon'
DEBUG = True

grid: BaseGrid = read_object(f'./dev/grid/{GAME}_grid.pkl')
border = 20

template_dir = Path(f'images/{GAME}/symbols/free_game')
img_dir = Path(f'images/{GAME}/screenshots/free_game')
output_dir = Path(f'temp/{GAME}_free_game_output')
output_dir.mkdir(parents=True, exist_ok=True)

scales = {}

for img_path in img_dir.iterdir():
    img_name = img_path.stem
    img = cv2.imread(str(img_path))
    if DEBUG and img_name != 'vlcsnap-2024-11-28-15h58m25s922':
        continue
    print(f'Processing {img_name}...')

    for i in range(grid.row):
        for j in range(grid.col):
            x, y, w, h = grid.get_roi(i, j)
            roi = img[y-border:y+h+border, x-border:x+w+border]

            best_match, best_score, num_matches = process_template_matches_gray(scales, template_dir, roi, [0.5, 1.3], 0.05, 0.5, border, DEBUG)
            grid[i, j] = best_match
            if DEBUG:
                print(f'Best match: {best_match}, best score: {best_score}, num matches: {num_matches}')

    output_path = output_dir / f'{img_name}.png'
    draw_bboxes_and_icons_on_image(img, template_dir, grid, str(output_path))

# img_path = Path('images/fu/screenshots/free_game/vlcsnap-2024-11-10-15h37m09s088.png')
# # img_path = Path('images/fu/screenshots/free_game/vlcsnap-2024-11-10-15h35m53s352.png')
# img_name = img_path.stem
# img = cv2.imread(str(img_path))

# for i in range(grid.row):
#     for j in range(grid.col):

#         x, y, w, h = grid.get_roi(i, j)
#         roi = img[y-border:y+h+border, x-border:x+w+border]
#         # roi = img[y:y+h, x:x+w]
        
#         best_match, best_scale, num_matches = process_template_matches_sift(template_dir, roi, (0.7, 1.3), True)
#         grid[i, j] = best_match
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
