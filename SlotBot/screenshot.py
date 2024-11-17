import pygetwindow as gw
import pyautogui
import os

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
