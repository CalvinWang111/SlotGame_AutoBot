import os
import cv2
from pathlib import Path
import json
def template_matching(game_screenshot, template, threshold=0.7, method=cv2.TM_CCOEFF_NORMED):
    # Read the main image and the template
    game_scene_gray = cv2.cvtColor(game_screenshot, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template.shape[:2]

    # Apply template matching
    res = cv2.matchTemplate(game_scene_gray, template_gray, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)
    # if max_val >= threshold:
    #     x, y = max_loc
    #     return [x, y, w, h]
    # else:
    #     return []
    return max_val >= threshold


def test_template_matching(game_screenshot_path, all_freegame_btn_json_path):
    root_dir = Path(__file__).parent.parent
    game_screenshot = cv2.imread(game_screenshot_path)
    display_game_screenshot = game_screenshot.copy()
    matched_loc = []
    all_freegame_btn_json = json.load(open(all_freegame_btn_json_path, mode='r', encoding='utf=8'))
    for key, value in all_freegame_btn_json.items():
        template = cv2.imread(value['path'])
        if template_matching(game_screenshot, template):
            matched_loc.append(value['contour'])
            x,y,w,h = value['contour']
            cv2.rectangle(display_game_screenshot, (int(x),int(y)), (int(x+w), int(y+h)), (255,255,255), 3, cv2.LINE_AA)
    saved_image_path = os.path.join(root_dir, 'images', 'matched_image.png')
    cv2.imwrite(saved_image_path, display_game_screenshot)
    return matched_loc


if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent
    game_screenshot_path = os.path.join(root_dir, 'images', 'FuXinfreegame.png')
    # template_path = os.path.join(root_dir, 'marquee_tool', 'FuXin_runtime', 'template', 'free_btn.png')
    all_freegame_btn_json_path = os.path.join(root_dir, 'marquee_tool', 'FuXin_runtime', 'FuXin_runtime_regions.json')
    all_freegame_btn_json = json.load(open(all_freegame_btn_json_path, encoding='utf=8'))
    game_screenshot = cv2.imread(game_screenshot_path)
    # template = cv2.imread(template_path)
    # loc = template_matching(game_screenshot, template)

    loc = test_template_matching(game_screenshot_path, all_freegame_btn_json)

    
