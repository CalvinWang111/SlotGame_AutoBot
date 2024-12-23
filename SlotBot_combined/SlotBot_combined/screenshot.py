import pygetwindow as gw 
import pyautogui 
import os 
from PIL import Image, ImageDraw 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 
# 使用 Matplotlib 的交互工具 
from matplotlib.widgets import RectangleSelector 
from tkinter import Tk, simpledialog 
import time
 
class GameScreenshot: 
    @staticmethod 
    def capture_screenshot(window_title,filename,region=None, np_return = False): 
        """取得遊戲畫面截圖""" 
        try: 
            # Find the window by title 
            window = gw.getWindowsWithTitle(window_title)[0]  # Get the first matching window 
            window.activate()  # Bring the window to the foreground if needed
            time.sleep(1)
             
            # Get window position and size 
            x, y, width, height = window.left, window.top, window.width, window.height 
            print(f"Window position and size: x={x}, y={y}, width={width}, height={height}") 
             
            # Capture the specified region 
            screenshot = pyautogui.screenshot(region=(x, y, width, height)) 

            if np_return:
                return np.array(screenshot)

            # Define the directory where the screenshot will be saved 
            save_directory = os.path.join('./images/')
                 
            # Create the directory if it does not exist 
            os.makedirs(save_directory, exist_ok=True) 
             
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
    def clickable(snapshot_path=None, snapshot_array=None, highest_confidence_images=None, target_buttons=None, draw_and_show=False):
        """
        Analyze the intensity of button regions in the snapshot to determine their clickable state.

        Args:
            snapshot_path (str): Path to the snapshot image.
            snapshot_array (np.ndarray): NumPy array of the snapshot image.
            highest_confidence_images (dict): A dictionary containing bounding box info for detected buttons.
            target_buttons (list): List of button names to analyze. If None, analyze all buttons.
            draw_and_show (bool): Whether to draw bounding boxes and display the image.

        Returns:
            avg_intensities (dict): A dictionary mapping each class_id to a list of average intensities.
        """
        # Define the label mapping
        label_map = {
                    0: "button_max_bet",
                    1: "button_additional_bet",
                    2: "button_close",
                    3: "confirm",
                    4: "button_decrease_bet",
                    5: "button_home",
                    6: "button_increase_bet",
                    7: "button_info",
                    8: "receive",
                    9: "button_speedup_spin",
                    10: "button_start_spin",
                    11: "button_three_dot",
                    12: "gold_coin",
                    13: "gold_ingot",
                    14: "stickers",
                }

        # Convert target_buttons to class_ids for efficient filtering
        target_class_ids = [class_id for class_id, name in label_map.items() if target_buttons is None or name in target_buttons]

        avg_intensities = {}  # To store average intensities for each class_id


        # Assuming `gray_img` is your grayscale image in NumPy array format
        def save_gray_image(gray_img, save_path="./output_gray.png"):
            try:
                # Ensure the image is in the correct range (0-255)
                if gray_img.max() > 255 or gray_img.min() < 0:
                    gray_img = (255 * (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())).astype(np.uint8)

                # Convert NumPy array back to a PIL image and save
                gray_image = Image.fromarray(gray_img.astype(np.uint8))
                gray_image.save(save_path)
                #print(f"Grayscale image saved at {os.path.abspath(save_path)}")
            except Exception as e:
                print(f"Error saving grayscale image: {e}")

        # Load the snapshot image
        try:
            if snapshot_array is not None:
                # Convert NumPy array to grayscale directly
                img = Image.fromarray(snapshot_array)
                gray_img = snapshot_array if len(snapshot_array.shape) == 2 else np.mean(snapshot_array, axis=2)
                #save_gray_image(gray_img=gray_img)
            elif snapshot_path is not None:
                # Load image from file path and convert to grayscale
                img = Image.open(snapshot_path)
                gray_img = np.array(img.convert("L"))
                
            else:
                raise ValueError("Either snapshot_path or snapshot_array must be provided.")

            if draw_and_show:
                draw = ImageDraw.Draw(img)

            for class_id, info in highest_confidence_images.items():
                if class_id not in target_class_ids:  # Skip non-target buttons
                    continue

                avg_intensities[class_id] = []  # Initialize list for each class_id

                if isinstance(info, list):  # Multiple bounding boxes for the class
                    for info_item in info:
                        (x, y, w, h) = info_item['contour']

                        # Crop the region specified by the bounding box
                        cropped_gray = gray_img[y:y + h, x:x + w]

                        # Calculate the average intensity of the grayscale button region
                        avg_intensity = cropped_gray.mean()
                        avg_intensities[class_id].append(avg_intensity)

                        if draw_and_show:
                            # Draw bounding box
                            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)

                else:  # Single bounding box for the class
                    (x, y, w, h) = info['contour']

                    # Crop the region specified by the bounding box
                    cropped_gray = gray_img[y:y + h, x:x + w]

                    # Calculate the average intensity of the grayscale button region
                    avg_intensity = cropped_gray.mean()
                    avg_intensities[class_id].append(avg_intensity)

                    #print('in screenshot clickable 142 lines','info', info)
                   # print('avg_intensity', avg_intensity, 'avg_intensities', avg_intensities)
                    if draw_and_show:
                        # Draw bounding box
                        draw.rectangle([x, y, x + w, y + h], outline="red", width=5)

            if draw_and_show:
                # Display the image with bounding boxes
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.show()

        except FileNotFoundError:
            print(f"Snapshot not found at {snapshot_path}")
        except ValueError as e:
            print(f"Error: {e}")

        return avg_intensities



    '''   
    @staticmethod 
    def clickable(snapshot_path, highest_confidence_images, draw_and_show=False): 
        """ 
        Analyze the intensity of button regions in the snapshot to determine their clickable state. 
 
        Args: 
            snapshot_path (str): Path to the snapshot image. 
            highest_confidence_images (dict): A dictionary containing bounding box info for detected buttons. 
            draw_and_show (bool): Whether to draw bounding boxes and display the image. 
 
        Returns: 
            avg_intensities (dict): A dictionary mapping each class_id to a list of average intensities. 
        """ 
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
 
        avg_intensities = {}  # To store average intensities for each class_id 
 
        # Load the snapshot image 
        try: 
            with Image.open(snapshot_path) as img: 
                draw = ImageDraw.Draw(img) if draw_and_show else None 
 
                for class_id, info in highest_confidence_images.items(): 
                    avg_intensities[class_id] = []  # Initialize list for each class_id 
 
                    if isinstance(info, list):  # Multiple bounding boxes for the class 
                        for info_item in info: 
                            (x, y, w, h) = info_item['contour'] 
 
                            # Crop the region specified by the bounding box 
                            cropped_img = img.crop((x, y, x + w, y + h)) 
 
                            # Convert the button region to grayscale to measure intensity 
                            button_gray = np.array(cropped_img).mean(axis=2)  # Convert RGB to grayscale 
 
                            # Calculate the average intensity of the grayscale button region 
                            avg_intensity = button_gray.mean() 
                            #avg_intensities[class_id].append(avg_intensity) 
 
                            if draw_and_show: 
                                # Draw bounding box 
                                draw.rectangle([x, y, x + w, y + h], outline="red", width=5) 
                    else:  # Single bounding box for the class 
                        (x, y, w, h) = info['contour'] 
 
                        # Crop the region specified by the bounding box 
                        cropped_img = img.crop((x, y, x + w, y + h)) 
 
                        # Convert the button region to grayscale to measure intensity 
                        button_gray = np.array(cropped_img).mean(axis=2)  # Convert RGB to grayscale 
 
                        # Calculate the average intensity of the grayscale button region 
                        avg_intensity = button_gray.mean() 
                        avg_intensities[class_id].append(avg_intensity) 
 
                        if draw_and_show: 
                            # Draw bounding box 
                            draw.rectangle([x, y, x + w, y + h], outline="red", width=5) 
 
                if draw_and_show: 
                    # Display the image with bounding boxes 
                    plt.figure(figsize=(10, 6)) 
                    plt.imshow(img) 
                    plt.axis('off') 
                    plt.show() 
 
        except FileNotFoundError: 
            print(f"Snapshot not found at {snapshot_path}") 
 
        return avg_intensities 
    ''' 

    @staticmethod 
    def intensity_check(initial_avg_intensities, avg_intensities, intensity_threshold): 
        """ 
        Compare the average intensities of images between two `clickable` calls and check if  
        the difference for any image in the same class exceeds the given threshold. 
 
        Args: 
            initial_avg_intensities (dict): The average intensities from the first `clickable` call. 
            avg_intensities (dict): The average intensities from the second `clickable` call. 
            intensity_threshold (float): The allowed intensity difference threshold. 
 
        Returns: 
            bool: True if all intensity differences are within the threshold, False otherwise. 
        """ 
        for class_id in initial_avg_intensities: 
            # Ensure both sets have the same class_id data 
            if class_id not in avg_intensities: 
                continue 
             
            # Compare intensities for the same class 
            initial_intensities = initial_avg_intensities[class_id] 
            current_intensities = avg_intensities[class_id] 
 
            #print('into screenshot.py intensity_check') 
            #print('intial_avg_intensities', initial_avg_intensities, type(initial_avg_intensities)) 
            #print('avg_intensities', avg_intensities, type(avg_intensities)) 
 
            # Check each pair of intensities 
            for i in range(min(len(initial_intensities), len(current_intensities))): 
                if abs(initial_intensities[i] - current_intensities[i]) >= intensity_threshold: 
                    return False  # Return False if any intensity exceeds the threshold 
 
        return True  # Return True if all intensity differences are within the threshold 
     
    @staticmethod 
    def interactive_labeling(image_path,Snapshot): 
        """ 
        交互式框選區域並添加標記，返回位置和標記數據。 
         
        Parameters: 
            image_path (str): 圖像文件的路徑。 
         
        Returns: 
            regions (dict): 包含按鍵名稱和框選區域的字典。 
                            格式為 {"button_name": (x, y, w, h)}。 
        """ 
            # 初始化 Tkinter 
        root = Tk() 
        root.withdraw()  # 隱藏主窗口 
        regions = {} 
        fig, ax = plt.subplots() 
        img = plt.imread(image_path) 
        ax.imshow(img) 
        plt.title("Drag to select a region. Close the window to finish.") 
        selected_rectangles = [] 
         
 
        def onselect(eclick, erelease): 
            """ 
            在用戶拖動鼠標選擇區域時觸發的事件。 
            """ 
 
            # Define the label mapping 
            label_map = {
                    0: "button_max_bet",
                    1: "button_additional_bet",
                    2: "button_close",
                    3: "confirm",
                    4: "button_decrease_bet",
                    5: "button_home",
                    6: "button_increase_bet",
                    7: "button_info",
                    8: "receive",
                    9: "button_speedup_spin",
                    10: "button_start_spin",
                    11: "button_three_dot",
                    12: "gold_coin",
                    13: "gold_ingot",
                    14: "stickers",
                }
            print('請依照框選區域，使用代號命名該按鍵功用',label_map) 
 
            x1, y1 = eclick.xdata, eclick.ydata 
            x2, y2 = erelease.xdata, erelease.ydata 
 
            # 確保坐標是正確的 (x, y, w, h)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
 

            # 使用 Tkinter 捕獲用戶輸入的按鍵名稱 
            button_class = simpledialog.askstring( 
            "Input", f"Enter button name for region ({x}, {y}, {w}, {h}):") 
 
            # 要求用戶輸入按鍵名稱 
            #button_class = input(f"Enter button name for region ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}): ").strip() 
             
            with Image.open(image_path) as img: 
                if button_class: 
                    # 裁剪指定區域 
                    cropped_img = img.crop((x, y, x + w, y + h)) 
 
                    # Define the directory where the screenshot will be saved 
                    output_dir = Snapshot+'/template' 
 
                    # Clear output directory if it exists, or create it if it doesn't 
                    if os.path.exists(output_dir): 
                        # Clear the output folder if it exists 
                        for filename in os.listdir(output_dir): 
                            file_path = os.path.join(output_dir, filename) 
                            try: 
                                if os.path.isfile(file_path) or os.path.islink(file_path): 
                                    os.unlink(file_path)  # Remove the file 
                                elif os.path.isdir(file_path): 
                                    os.rmdir(file_path)  # Remove the sub-directory (only works if empty) 
                            except Exception as e: 
                                print(f"Failed to delete {file_path}. Reason: {e}") 
 
                # Re-create the output folder 
                os.makedirs(output_dir, exist_ok=True) 
 
                # 保存裁剪圖像 
                cropped_img_path = os.path.join(output_dir, f"{button_class}.png") 
                cropped_img.save(cropped_img_path) 
                 
                # 填充 regions 字典 
                regions[button_class] = { 
                    'path': cropped_img_path, 
                    'confidence': 1,  # 默認信心值 
                    'contour': (x, y, w, h) 
                } 
 
                # 畫出選中的矩形 
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none') 
                ax.add_patch(rect) 
                selected_rectangles.append(rect) 
                plt.draw() 
 
 
        # 修正：移除 drawtype 
        toggle_selector = RectangleSelector(ax, onselect, interactive=True, button=[1], 
                                            minspanx=5, minspany=5, spancoords='pixels') 
 
        # 顯示圖像並等待用戶操作 
        plt.show() 
 
        # 關閉 Tkinter 
        root.destroy() 
 
        return regions