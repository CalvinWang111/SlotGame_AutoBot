import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from TemplateMatching.grid import BaseGrid
from TemplateMatching.utils import *
from TemplateMatching.symbol_recognizer import *

GAME = 'fu'

grid: BaseGrid = read_object(f'./dev/grid/{GAME}_grid.pkl')
template_scale = {}
border = 0

template_dir = Path('images/fu/symbols/free_game')
img_dir = Path('images/fu/screenshots/free_game')
output_dir = Path('temp/fu_free_game_output')
output_dir.mkdir(parents=True, exist_ok=True)

# for img_path in img_dir.iterdir():
#     img_name = img_path.stem
#     img = cv2.imread(str(img_path))
#     print(f'Processing {img_name}...')

#     for i in range(grid.row):
#         for j in range(grid.col):
#             x, y, w, h = grid.get_roi(i, j)
#             roi = img[y:y+h, x:x+w]
#             roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#             roi_gray = cv2.equalizeHist(roi_gray)

#             best_match, best_score, best_scale = find_best_match_gray(template_scale, template_dir, roi_gray, (1.0, 1.5), 0.05, 0.5, 20)
#             # template_scale = {}
#             grid[i, j] = best_match
            
#             # print(f'Best match for ({i}, {j}): {best_match} with score {best_score} and scale {best_scale}')
#             # print(template_scale)
#             # cv2.imshow('roi', roi)
#             # cv2.imshow('roi_gray', roi_gray)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()

#     output_path = output_dir / f'{img_name}.png'
#     draw_bboxes_and_icons_on_image(img, template_dir, grid, str(output_path))

img_path = Path('images/fu/screenshots/free_game/vlcsnap-2024-11-10-15h35m53s352.png')
# img_path = Path('images/fu/screenshots/free_game/vlcsnap-2024-11-10-15h35m47s222.png')
img_name = img_path.stem
img = cv2.imread(str(img_path))

for i in range(grid.row):
    for j in range(grid.col):

        x, y, w, h = grid.get_roi(i, j)
        roi = img[y-border:y+h+border, x-border:x+w+border]
        # roi = img[y:y+h, x:x+w]
        
        # best_match, best_score, best_scale = process_template_matches_gray(template_scale, template_dir, roi, (1.0, 1.5), 0.05, 0.2, 20)
        best_match, best_score, num_matches = process_template_matches_sift(template_dir, roi, 0.2)
        grid[i, j] = best_match
        
        print(f'Best match for ({i}, {j}): {best_match} with score {best_score}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
