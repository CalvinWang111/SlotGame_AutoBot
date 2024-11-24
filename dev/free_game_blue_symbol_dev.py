import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from TemplateMatching.grid import BaseGrid
from TemplateMatching.utils import *
from TemplateMatching.symbol_recognizer import *

def template_matching_gray(template_gray, mask, img_gray, scale_range, scale_step, threshold, border, match_one=False):
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    padding = 5  # Padding to add around the image for reverse matching
    
    matching_results = []  # To store the locations of matches
    
    for scale in scales:
        # Resize template and mask for the current scale
        resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
        resized_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
        result = None
        
        template_h, template_w = resized_template.shape[:2]
        img_h, img_w = img_gray.shape[:2]
        # Ensure the resized template is not larger than the image
        if template_h > img_h or template_w > img_w:
            if template_h >= img_h - 2 * border and template_w >= img_w - 2 * border:
                # perform reverse matching
                img_without_border = img_gray[border+padding:img_h-border-padding, border+padding:img_w-border-padding]
                result = cv2.matchTemplate(resized_template, img_without_border, cv2.TM_CCORR_NORMED)
            else:
                continue
        else:
            # Perform template matching
            result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCORR_NORMED, mask=resized_mask)
        
        # Find locations where the match is greater than the threshold
        loc = np.where(result >= threshold)

        # Collect all the matching points
        for pt in zip(*loc[::-1]):  # Switch x and y in zip
            matching_results.append((pt, scale, result[pt[1], pt[0]])) # (top_left, scale, match_val)
    return matching_results

def find_best_match_gray(template_scale, template_dir, target_roi, scale_range, scale_step, threshold, border):
    best_match = None
    best_score = None
    best_scale = None

    # Iterate through each template in the directory
    for template_path in Path(template_dir).iterdir():
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.equalizeHist(template_gray)
        _, _, _, alpha_channel = cv2.split(template)
        mask = cv2.threshold(alpha_channel, 16, 255, cv2.THRESH_BINARY)[1]
        template_name = template_path.stem
        
        if template_name in template_scale:
            scale = template_scale[template_name]
            matching_results = template_matching_gray(template_gray, mask, target_roi, (scale, scale), scale_step, threshold, border, True)
        else:
            matching_results = template_matching_gray(template_gray, mask, target_roi, scale_range, scale_step, threshold, border, True)
            
        

        # Extract the best score
        if matching_results:
            top_result = max(matching_results, key=lambda x: x[2])  # x[2] should be the score
            top_score = top_result[2]
            if best_score is None or top_score > best_score:
                best_match = template_path.stem
                best_score = top_score
                best_scale = top_result[1]
                
    if not (template_name in template_scale):
        template_scale[best_match] = best_scale
    
    if not best_match:
        return None, None, None
    
    return best_match, best_score, best_scale

grid: BaseGrid = read_object('./dev/grid.pkl')
template_scale = {}

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

img_path = Path('images/fu/screenshots/free_game/vlcsnap-2024-11-10-15h35m47s222.png')
img_name = img_path.stem
img = cv2.imread(str(img_path))

for i in range(grid.row):
    for j in range(grid.col):
        if (i != 2 or j != 4):
            continue
        x, y, w, h = grid.get_roi(i, j)
        roi = img[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.equalizeHist(roi_gray)
        
        best_match, best_score, best_scale = find_best_match_gray(template_scale, template_dir, roi_gray, (1.0, 1.5), 0.05, 0.5, 20)
        grid[i, j] = best_match
        print(f'Best match for ({i}, {j}): {best_match} with score {best_score} and scale {best_scale}')
        
        matched_template = cv2.imread(str(template_dir / f'{best_match}.png'))
        matched_template_gray = cv2.cvtColor(matched_template, cv2.COLOR_BGR2GRAY)
        matched_template_gray = cv2.equalizeHist(matched_template_gray)
        cv2.imshow('matched_template_gray', matched_template_gray)
        cv2.imshow('roi', roi)
        cv2.imshow('roi_gray', roi_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
