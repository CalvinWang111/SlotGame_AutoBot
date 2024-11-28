from screenshot import GameScreenshot
import pygetwindow as gw
import pyautogui
import time

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

    def Windowcontrol(self, highest_confidence_images, classId):
        
        for item in highest_confidence_images.items():
            if item[0] == classId:
                if classId in [12, 13]:
                    for subitem in item[1]:
                        x,y,w,h = subitem['contour']
                        x_offset = x + w / 2
                        y_offset = y + h / 2
                        
                        print('special',classId,x_offset, y_offset)
                        self.click_in_window('BlueStacks App Player',x_offset, y_offset)
                else:
                    x,y,w,h = item[1]['contour']
                    x_offset = x + w / 2
                    y_offset = y + h / 2

                    print(x_offset, y_offset)
                    self.click_in_window('BlueStacks App Player',x_offset, y_offset)