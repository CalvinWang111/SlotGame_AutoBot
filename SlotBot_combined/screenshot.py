import pygetwindow as gw
import pyautogui
import os
from PIL import Image
import numpy as np

class GameScreenshot:
    @staticmethod
    def capture_screenshot(window_title,filename,region=None):
        """取得遊戲畫面截圖"""
        try:
            # Define the directory where the screenshot will be saved
            save_directory = './images/'
                
            # Create the directory if it does not exist
            os.makedirs(save_directory, exist_ok=True)
            
            # Find the window by title
            window = gw.getWindowsWithTitle(window_title)[0]  # Get the first matching window
            # window.activate()  # Bring the window to the foreground if needed
            
            # Get window position and size
            x, y, width, height = window.left, window.top, window.width, window.height
            print(f"Window position and size: x={x}, y={y}, width={width}, height={height}")
            
            # Capture the specified region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # Save the screenshot to the specified file
            full_path = os.path.join(save_directory, filename + '.png')
            screenshot.save(full_path)
            print(f"Screenshot saved as {full_path}")
        except IndexError:
            print(f"Window titled '{window_title}' not found.")

    @staticmethod
    def click(position):
        """模擬點擊指定位置"""
        pyautogui.click(position)

    @staticmethod
    def move_to(position):
        """移動滑鼠到指定位置"""
        pyautogui.moveTo(position)

    @staticmethod
    def clickable(snapshot_path, highest_confidence_images):
        
        # Define the label mapping
        label_map = {
            0: "button_max_bet",
            1: "button_additional_bet",
            2: "button_close",
            3: "button_decrease_bet",
            4: "button_home",
            5: "button_increase_bet",
            6: "button_info",
            7: "button_speedup_spin",
            8: "button_start_spin",
            9: "button_three_dot",
            10: "gold_coin",
            11: "gold_ingot",
            12: "stickers"
        }
        '''
        # Clear the output folder if it exists
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)  # Remove the sub-directory (only works if empty)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        # Re-create the output folder
        os.makedirs(output_folder, exist_ok=True)
        '''
        # Load the snapshot image
        try:
            with Image.open(snapshot_path) as img:
                for class_id, info in highest_confidence_images.items():
                    if class_id==8:
                        (x, y, w, h) = info['contour']
                    
                        # Crop the region specified by the bounding box
                        cropped_img = np.array(img)[y:y+h,x:x+w,0:3]

                        # Convert the button region to grayscale to measure intensity
                        button_gray = np.mean(cropped_img[:,:,0:3], axis=2)  # Using RGB-to-grayscale mean approximation
                            
                        # Calculate the average intensity of the grayscale button region
                        avg_intensity = np.mean(button_gray)

                        '''
                        # Save the cropped image
                        cropped_filename = os.path.join(output_folder, f'{label_map[class_id]}.png')
                        cropped_img.save(cropped_filename)
                        print("Saved segment"+ label_map[class_id] + f" to {cropped_filename}")
                        '''
                        return avg_intensity
        except FileNotFoundError:
            print(f"Snapshot not found at {snapshot_path}")

        
    
    @staticmethod
    def clickable_np(img, highest_confidence_images):
        # Load the snapshot image

        for class_id, info in highest_confidence_images.items():
            if class_id==8:
                (x, y, w, h) = info['contour']
            
                # Crop the region specified by the bounding box
                cropped_img = img[y:y+h,x:x+w,0:3]

                # Convert the button region to grayscale to measure intensity
                button_gray = np.mean(cropped_img[:,:], axis=2)  # Using RGB-to-grayscale mean approximation
                    
                # Calculate the average intensity of the grayscale button region
                avg_intensity = np.mean(button_gray)

                '''
                # Save the cropped image
                cropped_filename = os.path.join(output_folder, f'{label_map[class_id]}.png')
                cropped_img.save(cropped_filename)
                print("Saved segment"+ label_map[class_id] + f" to {cropped_filename}")
                '''
                return avg_intensity