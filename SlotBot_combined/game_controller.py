from screenshot import GameScreenshot
import pygetwindow as gw
import pyautogui
import time
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from TemplateMatching.template_matching import test_template_matching
from pathlib import Path
import random
import json
import os
import cv2
import numpy as np

class GameController:
    @staticmethod
    def click_in_window(window_title, x_offset, y_offset, clicks=1, interval=0.0, button='left'):
        """
        Click at a specific location within a window identified by its title.
        
        Parameters:
        - window_title (str): The title of the window to interact with.
        - x_offset (int): The x offset within the window where the click should occur.
        - y_offset (int): The y offset within the window where the click should occur.
        - clicks (int): Number of clicks to perform (default is 1).
        - interval (float): Interval between clicks if multiple clicks are specified.
        - button (str): Which mouse button to click, 'left', 'right', or 'middle' (default is 'left').
        """
        try:
            # Find the window by title
            window = gw.getWindowsWithTitle(window_title)[0]  # Assumes the first match
            #window.activate()  # Bring the window to the foreground
            time.sleep(0.5)  # Wait for the window to come to the front
            
            # Get the window's position and size
            x, y = window.left, window.top
            
            # Calculate the absolute click position based on the window position
            click_x = x + x_offset
            click_y = y + y_offset
            
            # Move and click at the calculated position
            pyautogui.click(click_x, click_y, clicks=clicks, interval=interval, button=button)
            
            print(f"Clicked at ({click_x}, {click_y}) in window '{window_title}'")
            
        except IndexError:
            print(f"Window titled '{window_title}' not found.")

    def auto_click_from_center(window_title, x_offset, y_offset, regions, step=4, clicks=1, interval=0.0, button='left'):
        """
        Automatically click from the center of the window outward until the distance is less than twice the offsets,
        while avoiding clicking below the highest contour point.

        Parameters:
        - window_title (str): The title of the window to interact with.
        - classId (any): Identifier for the class of interest.
        - highest_confidence_images (dict): Dictionary containing contour data for the class.
        - x_offset (int): Initial x offset for clicking.
        - y_offset (int): Initial y offset for clicking.
        - step (int): Step size to increment offsets outward (default is 4).
        - clicks (int): Number of clicks to perform at each position (default is 1).
        - interval (float): Interval between clicks (default is 0.0).
        - button (str): Which mouse button to click, 'left', 'right', or 'middle' (default is 'left').
        """
        try:
            # Get contour data
            x, y, w, h = regions['temp']['contour']
            print('regions (x, y, w, h)', x, y, w, h)
            contour_top = y

            # Find the window by title
            window = gw.getWindowsWithTitle(window_title)[0]  # Assumes the first match

            # Get the window's position and size
            width, height = window.width, window.height
            center_x_offset = width // 2
            center_y_offset = height // 2

            current_x_offset, current_y_offset = 0, 0

            while abs(current_x_offset) < (width // 2 - step) and abs(current_y_offset) < (height // 2 - step):
                # Calculate the absolute click position
                absolute_y = center_y_offset + current_y_offset

                # Avoid clicking below the contour's highest point
                if absolute_y < contour_top:
                    GameController.click_in_window(window_title, center_x_offset + current_x_offset, absolute_y, clicks, interval, button)

                # Increment offsets outward
                current_x_offset += x_offset
                current_y_offset += y_offset

                if abs(current_x_offset) >= (width // 2 - step) or abs(current_y_offset) >= (height // 2 - step):
                    break

            print("Auto-clicking process completed.")

        except IndexError:
            print(f"Window titled '{window_title}' not found.")



    def Windowcontrol(self, highest_confidence_images, classId):
        for item in highest_confidence_images.items():
            if item[0] == classId:
                x,y,w,h = item[1]['contour']
                x_offset = x + w / 2
                y_offset = y + h / 2

                print(x_offset, y_offset)
                self.click_in_window('BlueStacks App Player',x_offset, y_offset)

    def sift_with_ransac(game_screenshot_path, all_freegame_btn_json_path):
        # Read images
        game_screenshot = cv2.imread(game_screenshot_path)
        matched_loc = []

        # Load JSON configuration
        with open(all_freegame_btn_json_path, mode='r', encoding='utf-8') as file:
            all_freegame_btn_json = json.load(file)

        # Get the original size of the game screenshot from JSON
        original_image_size = all_freegame_btn_json.get("original_image_size", {})
        original_width = original_image_size.get("width", game_screenshot.shape[1])
        original_height = original_image_size.get("height", game_screenshot.shape[0])

        # Calculate scaling factors
        scale_x = game_screenshot.shape[1] / original_width
        scale_y = game_screenshot.shape[0] / original_height

        for key, value in all_freegame_btn_json.get("regions", {}).items():
            # Load template image
            template_img = cv2.imread(value['path'], cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"Template image {value['path']} not found.")
                continue

            # Compute relative position and scale it to current screenshot
            contour = value.get("contour", [])
            if len(contour) == 4:
                x, y, w, h = contour
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                w_scaled = int(w * scale_x)
                h_scaled = int(h * scale_y)

                # Crop region of interest (ROI) from game screenshot
                roi = game_screenshot[y_scaled:y_scaled + h_scaled, x_scaled:x_scaled + w_scaled]
                if roi is None or roi.size == 0:
                    print(f"ROI for {key} is empty or invalid.")
                    continue

                # Preprocess images (Gaussian Blur and CLAHE)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                template_img = cv2.GaussianBlur(template_img, (5, 5), 0)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_gray = clahe.apply(roi_gray)
                template_img = clahe.apply(template_img)

                # Initialize SIFT detector
                sift = cv2.SIFT_create()

                # Detect and compute keypoints and descriptors
                keypoints1, descriptors1 = sift.detectAndCompute(template_img, None)
                keypoints2, descriptors2 = sift.detectAndCompute(roi_gray, None)

                # 检查描述符是否为 None 或者长度为 0
                if descriptors1 is None or descriptors2 is None:
                    print(f"Skipping {key} due to empty descriptors.")
                    continue

                # 转换描述符为 float32 类型（FLANN 需要此格式）
                descriptors1 = np.asarray(descriptors1, dtype=np.float32)
                descriptors2 = np.asarray(descriptors2, dtype=np.float32)

                # Match descriptors using FLANN-based matcher
                index_params = dict(algorithm=1, trees=20)  # FLANN KDTree Index
                search_params = dict(checks=20)  # Number of checks
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(descriptors1, descriptors2, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.3 * n.distance:
                        good_matches.append(m)

                # Minimum number of matches to consider it valid
                MIN_MATCH_COUNT = 150
                if len(good_matches) >= MIN_MATCH_COUNT:
                    # Extract matched keypoints
                    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Compute Homography using RANSAC
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
                    matches_mask = mask.ravel().tolist() if mask is not None else []

                    if M is not None:
                        h, w = template_img.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # Bounding box for matched region
                        x, y, w, h = cv2.boundingRect(dst)
                        matched_loc.append((x_scaled + x, y_scaled + y, w, h))

                        # Draw matched region on the game screenshot
                        game_screenshot = cv2.polylines(game_screenshot, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # Visualize matches
                match_img = cv2.drawMatches(
                    template_img, keypoints1,
                    roi_gray, keypoints2,
                    good_matches, None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                )
                #cv2.imshow(f"Matches for {key}", match_img)

        #cv2.imshow("Matched Regions", game_screenshot)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return matched_loc




    def freegame_control_NoVIT(root_dir=Path(__file__).parent.parent,window_name='BlueStacks App Player', Snapshot=''):
        screenshot = GameScreenshot()
        vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
        sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
        sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
        #sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
        images_dir = os.path.join(root_dir, 'images')
        success_continue = False
        all_freegame_btn_json_path = os.path.join(root_dir, 'marquee_tool', Snapshot, Snapshot + '_regions.json')

        if not all_freegame_btn_json_path is None:
            try:
                
                print('into fg try loop with no VIT')
                screenshot.capture_screenshot(window_title=window_name, images_dir=images_dir, filename=Snapshot+'freegame')
                freegame_screenshot = os.path.join(root_dir, 'images', Snapshot+'freegame.png')
                
                #try SIFT
                print('SIFT matching with fg btn json')
                sift_matched_loc = GameController.sift_with_ransac(freegame_screenshot, all_freegame_btn_json_path)
                
                # try template
                print('Template Matching matching with fg btn json')
                template_matched_loc = test_template_matching(freegame_screenshot, all_freegame_btn_json_path)

                # Combine locations from both methods
                all_matched_loc = sift_matched_loc + template_matched_loc
                #all_matched_loc = template_matched_loc
                print('all_matched_loc', all_matched_loc)

                if len(all_matched_loc) > 1:
                    # Randomly choose a location from combined results
                    loc = random.choice(all_matched_loc)
                    GameController.click_in_window(
                        window_title=window_name, 
                        x_offset=loc[0] + loc[2] // 2, 
                        y_offset=loc[1] + loc[3] // 2
                    )
                    success_continue = True
                elif len(all_matched_loc) == 1:
                    # Use the only matched location
                    loc = all_matched_loc[0]
                    GameController.click_in_window(
                        window_title=window_name, 
                        x_offset=loc[0] + loc[2] // 2, 
                        y_offset=loc[1] + loc[3] // 2
                    )
                    success_continue = True
                else:
                    # No matches found
                    print("Unable to process. No matches found.")
                    success_continue = False
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            success_continue = True
        
        return success_continue

    def freegame_control(root_dir=Path(__file__).parent.parent,window_name='BlueStacks App Player', Snapshot=''):
        screenshot = GameScreenshot()
        vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
        sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
        sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
        #sam_model_cfg = os.path.join(root_dir, 'sam2_configs', 'sam2_hiera_l.yaml')
        images_dir = os.path.join(root_dir, 'images')
        success_continue = False

        try:
            print('into try loop')
            screenshot.capture_screenshot(window_title=window_name, images_dir=images_dir, filename=Snapshot+'freegame')
            
            freegame_screenshot = os.path.join(root_dir, 'images', Snapshot+'freegame.png')
            all_freegame_btn_json_path = os.path.join(root_dir, 'marquee_tool', Snapshot, Snapshot + '_regions.json')
            
            # try SIFT
            sift_matched_loc = GameController.sift_with_ransac(freegame_screenshot, all_freegame_btn_json_path)
            
            # try template
            template_matched_loc = test_template_matching(freegame_screenshot, all_freegame_btn_json_path)
            # Combine locations from both methods
            all_matched_loc = sift_matched_loc + template_matched_loc

            if len(all_matched_loc) > 1:
                # Randomly choose a location from combined results
                loc = random.choice(all_matched_loc)
                GameController.click_in_window(
                    window_title=window_name, 
                    x_offset=loc[0] + loc[2] // 2, 
                    y_offset=loc[1] + loc[3] // 2
                )
                success_continue = True
            elif len(all_matched_loc) == 1:
                # Use the only matched location
                loc = all_matched_loc[0]
                GameController.click_in_window(
                    window_title=window_name, 
                    x_offset=loc[0] + loc[2] // 2, 
                    y_offset=loc[1] + loc[3] // 2
                )
                success_continue = True
            else:
                # No matches found
                print("Unable to process. No matches found.")
                success_continue = False

            #try VIT model
            maskDict_path = os.path.join(images_dir, Snapshot+'freegame' + ".png")
            sam = SAMSegmentation(Snapshot=Snapshot,images_dir=images_dir, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)
            maskDict = sam.segment_image(maskDict_path)
                
            #Classify components
            vit = ViTRecognition(
                     Snapshot=Snapshot, 
                     maskDict=maskDict, 
                     model_path=vit_model_path
            )
            highest_confidence_images, template_folder = vit.classify_components()
            vit.output_json(template_folder=template_folder, highest_confidence_images=highest_confidence_images)
                
            # Check for specific predictions
            if any(key in [3, 8, 12, 13] for key in highest_confidence_images.keys()):
                key = [key for key in highest_confidence_images.keys() if key in [3, 8, 12, 13] ]
                print('detecting 金元寶或金幣')
                GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=key[0])
                success_continue = True
        except Exception as e:
            print(f"An error occurred: {e}")
        
        return success_continue
    
