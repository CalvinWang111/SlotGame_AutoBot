import os
import cv2
def template_matching(game_screenshot, template, threshold=1, method=cv2.TM_CCOEFF_NORMED):
    # Read the main image and the template
    game_scene_gray = cv2.cvtColor(game_screenshot, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template.shape[:2]

    # Apply template matching
    res = cv2.matchTemplate(game_scene_gray, template_gray, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        x, y = max_loc
        return [x, y, w, h]
    else:
        return []

def test_template_matching(game_screenshot, all_template_path):
    pass

if __name__ == '__main__':
    pass