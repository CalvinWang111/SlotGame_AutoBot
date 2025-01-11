import cv2
import sys
from pathlib import Path
import numpy as np
import os
import math
import pygetwindow as gw
import mss
import time
from queue import Queue
import threading
import json
from Symbol_recognition.grid import BullGrid
from screenshot import GameScreenshot
from game_controller import GameController
from paddleocr import PaddleOCR

target_fps = 30
MAX_BUFFER_SIZE = 32
DEBUG = False

class StoppingFrameCapture:
    def __init__(self,window_name,save_dir, Snapshot, elapsed_time_threshold,game_name,config_file):
        self.window_name = window_name
        self.save_dir = save_dir
        self.__output_counter = 0
        self.free_gamestate = False
        self.processfail = False
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = 0      
        self.Snapshot = Snapshot
        self.time_threshold = elapsed_time_threshold
        self.frame_buffer = Queue()
        self.keywords = ['開始旋轉','自動旋轉','長按','開始']
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

        self.pause_event = threading.Event()  # 控制線程的事件
        self.pause_event.set()  # 初始化為未暫停狀態
        if(game_name=="bull"):
            self.bull_mode = True
        else:
            self.bull_mode = False

        self.config:json = self.load_config(config_file)
        self.use_key_frame = self.config.get('use_key_frame', False)
        self.timing_offset = self.config.get('timing_offset', 0)
        self.feature_sensitivity = self.config.get('feature_sensitivity', 1)
        self.optical_flow_fineness = self.config.get('optical_flow_fineness', 1)
        self.moving_distance_rate = self.config.get('moving_distance_rate', 1)
        self.use_upward_flow = self.config.get('use_upward_flow', False)
        self.strict_mode = self.config.get('optical_flow_strict_mode', False)
        

    @staticmethod
    def load_config(config_file: Path):
        with config_file.open("r") as file:
            return json.load(file)

    def __get_window_frame(self,frame_buffer):
            window = gw.getWindowsWithTitle(self.window_name)[0]
            left, top, width, height = window.left, window.top, window.width, window.height
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct = mss.mss()
            frame_time = 1/target_fps
            count = 1
            
            while not self.__terminated:
                self.pause_event.wait()  # 等待事件被設置
                frame_start_time = time.time() 
                frame = np.array(sct.grab(monitor))

                if frame_buffer.qsize() < MAX_BUFFER_SIZE:
                    frame_buffer.put(frame)
                else:
                    print("Warning: Frame buffer is full")
                
                #avg_intensities = screenshot.clickable(snapshot_path='./images/'+self.Snapshot+'_runtime.png',highest_confidence_images=highest_confidence_images)
                '''
                clickable_start_time = time.time()
                avg_intensities = screenshot.clickable(snapshot_array=frame, highest_confidence_images=highest_confidence_images, target_buttons=["button_start_spin"])
                clickable_end_time = time.time()
                #print(f'screenshot elapsed time: {screenshot_end_time-screenshot_start_time}')
                #print(f'clickable elapsed time: {clickable_end_time-clickable_start_time}')

                intensity_start_time = time.time()
                if screenshot.intensity_check(initial_avg_intensities=intial_intensity, avg_intensities=avg_intensities, intensity_threshold=intensity_threshold):
                    self.__button_available = True
                else:
                    self.__button_available = False
                intensity_end_time = time.time()
                #print(f'Intensity check time: {intensity_end_time-intensity_start_time}')
                '''
                frame_elapsed = time.time() - frame_start_time
                if DEBUG:
                    print(f"Frame read time: {frame_elapsed}, Buffer size: {frame_buffer.qsize()}")
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)
                # else:
                #     print("Warning: Capture speed lower than target frame rate")
                count += 1
            sct.close()

    def get_key_frames(self, grid, intial_intensity,intensity_threshold,highest_confidence_images,save_images):
        """
        Get the frame at the moment the wheel stops
        """
        key_image_pathes = []

        def __detect_stopping_frame(self:StoppingFrameCapture,frame_buffer):
            roi_x, roi_y, roi_w, roi_h = grid.bbox
            sh = grid.symbol_height
            sw = grid.symbol_width
            # adjust detecting area into 3 * 5, whitch can make things easy
            if self.bull_mode:
                roi_h = 3*sh
                if self.free_gamestate==False:
                    roi_y += (grid.row - 3)*sh

            # setting Shi-Tomasi
            feature_params = dict(maxCorners=50000, qualityLevel=0.01*self.feature_sensitivity, minDistance=20, blockSize=20)

            # setting Lucas-Kanade optical flow
            lk_params = dict(winSize=(max(int(sh/7*self.optical_flow_fineness),3), max(int(sw/10),3)), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))


            # setting my parameter
            rolling_record_size = 9
            min_moving_down_distance = sh/5*self.moving_distance_rate
            min_point_number = min(int(roi_w*roi_h/100000+1),10)
            max_error = 25
            min_rolling_frames = 15
            max_degree = 5

            max_tan = math.tan(math.radians(max_degree))
            rolling_record = [False]
            rolling_frames = 0
            is_first = True
            last_capture_time = time.time()
            offset_counter = -1

            # some thing about arrow detection
            arrow_rolling_point_number = 0
            arrow_flag = False
            noice_count = 0
            max_noice = 1
            last_capture_frame = -999
            frame_number = 0
            arrow_combo = 0
            elapsed_start_time = time.time()
            screenshot = GameScreenshot()
            
            while not (self.__terminated==True and frame_buffer.qsize()==0):
                if not frame_buffer.empty():
                    self.pause_event.wait()  # 等待事件被設置
                    start_time = time.time()
                    frame = frame_buffer.get()
                    if is_first:
                        old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
                        is_first = False
                        continue
                    
                    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
                    rolling_now = False
                    rolling_point_number = 0
                    if(len(rolling_record)==rolling_record_size):
                        rolling_record.pop(0)

                    # get Optical Flow
                    p0 = cv2.goodFeaturesToTrack(new_frame, mask=None, **feature_params)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)
                    horizontal_range = [9999,-1]

                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        # get the points moving down
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            if err[i]<max_error:
                                a, b = int(new.ravel()[0])+roi_x,int(new.ravel()[1])+roi_y
                                c, d = int(old.ravel()[0])+roi_x,int(old.ravel()[1])+roi_y
                                dx, dy = a - c, b - d
                                if (dy >= min_moving_down_distance or self.use_upward_flow and dy <= -min_moving_down_distance) and abs(dx/dy) <= max_tan:                                    
                                    rolling_point_number += 1
                                elif self.strict_mode and dx>2:
                                    rolling_point_number -= 1

                                # arrow detection
                                if self.bull_mode == True and 8 <= frame_number-last_capture_frame <= 25:
                                    if noice_count <= max_noice:
                                        video_height = frame.shape[0]
                                        if video_height*0.004 < dx**2+dy**2 < video_height*2.8:
                                            horizontal_range[0] = min(horizontal_range[0],a,c)
                                            horizontal_range[1] = max(horizontal_range[1],a,c)
                                            if video_height*0.004 < abs(dy) < video_height*0.02 and abs(dx/dy) <= max_tan:
                                                arrow_rolling_point_number += 1
                    if rolling_point_number >= min_point_number:
                        rolling_now = True
                                    
                    # more arrow detection
                    if self.bull_mode == True and 8 <= frame_number-last_capture_frame <= 25:
                        points_width = horizontal_range[1]-horizontal_range[0]
                        if(DEBUG):
                            print(points_width)
                        if self.bull_mode == True and points_width <= 2*sw and points_width > 0 and arrow_rolling_point_number>0:
                            arrow_flag = True
                            rolling_now = True
                        elif points_width > 2*sw:
                            noice_count += 1
                    else:
                        if DEBUG and frame_number-last_capture_frame == 26:
                            print(arrow_flag,noice_count,arrow_combo)
                        noice_count = 0
                        arrow_flag = False

                    rolling_record.append(rolling_now)
                    old_frame = new_frame.copy()
                    
                    if(len(rolling_record)==rolling_record_size):
                        if True in rolling_record:
                            if ((rolling_record.index(True) == 0 and rolling_record.count(True) == 1) or frame_number-last_capture_frame==25) and (rolling_frames >= min_rolling_frames or (arrow_flag==True and noice_count <= max_noice and arrow_combo<5)):
                                # the wheel is stop right now
                                if(offset_counter<0):
                                    offset_counter = 0
                            else:
                                if frame_number-last_capture_frame>25:
                                    # the wheel is still rolling
                                    rolling_frames+=1
                                else:
                                    rolling_frames = 0
                        else:
                            rolling_frames = 0
                        if(offset_counter == max(0,self.timing_offset)):
                            # this is the moment when the wheel is stopped
                            if save_images:
                                cv2.imwrite(f"{self.save_dir}\\key_frame{self.__output_counter}.png", frame)
                                key_image_pathes.append(f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                            if arrow_flag:
                                print("Get arrow stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                if(DEBUG):
                                    print(arrow_flag,noice_count,arrow_combo)
                                arrow_combo += 1
                            else:
                                if save_images:
                                    print("Get stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                else:
                                    print("Get stopping frame")
                                arrow_combo = 0
                            self.__output_counter+=1
                            last_capture_time = time.time()
                            last_capture_frame = frame_number

                            avg_intensities = screenshot.clickable(snapshot_array=frame, highest_confidence_images=highest_confidence_images, target_buttons=["button_start_spin"])
                            if screenshot.intensity_check(initial_avg_intensities=intial_intensity, avg_intensities=avg_intensities, intensity_threshold=intensity_threshold):
                                self.__button_available = True
                            else:
                                self.__button_available = False
                            offset_counter = -1
                        elif(offset_counter>=0):
                            offset_counter += 1
                    if DEBUG:
                        print(f"Stopping-judgment time: {time.time()-start_time}, Buffer size: {frame_buffer.qsize()}")
                    
                    frame_number += 1
                
                # prevent busy waiting
                else:
                    time.sleep(0.01) 

                #Free Game state
                elapsed__time = (time.time() - elapsed_start_time)

                # Perform checks if elapsed time exceeds 10 seconds and is a multiple of 5
                if elapsed__time >= 10 and (int(elapsed__time) - 10) % 5 == 0:
                    avg_intensities = screenshot.clickable(snapshot_array=frame, highest_confidence_images=highest_confidence_images, target_buttons=["button_start_spin"])
                    if screenshot.intensity_check(initial_avg_intensities=intial_intensity, avg_intensities=avg_intensities, intensity_threshold=intensity_threshold):
                        self.__button_available = True
                    else:
                        self.__button_available = False
                        
                    if self.__button_available == False:
                        print('into freegame_control')
                        print('button state', self.__button_available)
                        self.pause_event.clear()  # 暫停線程

                         #檢查開始選轉按紐，顯示內容是否為開始旋轉，如果不是判定進入免費遊戲
                        (x, y, w, h) = highest_confidence_images[10]['contour']
                        ocr_result = self.ocr.ocr(frame[y:y + h, x:x + w], cls=True)
                        ocr_result = ocr_result[0]
                                        
                        # Extract text and confidence
                        results = [(item[1][0], item[1][1]) for item in ocr_result]

                        # Find the result with the highest confidence
                        highest_score_result = max(results, key=lambda x: x[1])

                        # Check if the highest score answer is "開始旋轉"
                        if any(keyword in highest_score_result[0] for keyword in self.keywords):
                            print("The spin button showing : '開始旋轉'.")
                            self.free_gamestate = False
                        else:
                            print("The spin button showing is not : '開始旋轉'.")

                            # 提取數字
                            numbers = [int(part) for text, _ in results for part in text.split('/') if part.isdigit()]

                            if len(numbers) >= 2:
                                print(f"Remaining Spins: {numbers[0]}, Free Games Won: {numbers[1]}")
                            else:
                                print("Could not extract sufficient numerical data.")
                            self.free_gamestate = True
                            success_continue = GameController.freegame_control(Snapshot=self.Snapshot)
                        self.pause_event.set()  # 恢復線程
                        #print('freegame control success to contunue: ', success_continue)
                elif elapsed__time > 30:
                    print('Slotgame AutoBot fail to process')
                    self.__terminated = True
                    self.processfail = True

                # Termination condition
                if self.__button_available==True:
                    if time.time()-last_capture_time>5 :
                        self.__terminated = True
                
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = time.time()
        self.frame_buffer.queue.clear()

        capture_thread = threading.Thread(target=self.__get_window_frame, args=(self.frame_buffer,))
        process_thread = threading.Thread(target=__detect_stopping_frame, args=(self,self.frame_buffer))

        capture_thread.start()
        process_thread.start()

        capture_thread.join()
        process_thread.join()

        print("round over")
        return(key_image_pathes)
    
    def get_static_frame(self,grid,images_dir, filename, duration=3):
        """
        When the wheel is stopped, call it to get the least disturbed frame by special effects.
        duration: in seconds, assume fps=30
        """

        def __detect_static_frame(self:StoppingFrameCapture,frame_buffer):
            roi_x, roi_y, roi_w, roi_h = grid.bbox
            sampling_interval = int(roi_w* roi_h/10000)
            sampling_interval = 5

            best_frame = None
            min_difference = 999999999
            is_first = True
            frame_number = 0

            while frame_number <= duration*30:
                fps_time = time.time()
                frame = frame = frame_buffer.get()               
                new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
                pixel_difference = 0

                if not is_first:
                    # Use NumPy's stride feature to sample rows and columns
                    rows = np.arange(0, new_frame.shape[0], sampling_interval)
                    cols = np.arange(0, new_frame.shape[1], sampling_interval)

                    # Create a grid to get all indices
                    row_indices, col_indices = np.meshgrid(rows, cols, indexing='ij')

                    # Adjust column indices (col_indices) with the offset
                    col_indices = (col_indices + (row_indices % sampling_interval)) % new_frame.shape[1]

                    # Extract pixel values corresponding to the indices
                    new_pixels = new_frame[row_indices, col_indices]
                    old_pixels = old_frame[row_indices, col_indices]

                    # Compute pixel differences and sum them up
                    pixel_difference = np.sum(np.abs(new_pixels - old_pixels))

                    if min_difference > pixel_difference:
                        min_difference = pixel_difference
                        best_frame = frame.copy()
                        if pixel_difference==0:
                            break

                old_frame = new_frame.copy()
                is_first = False

                # print("fps:",1/(time.time()-fps_time),"\t min difference:",pixel_difference)
                frame_number += 1
            
            self.__terminated = True
            if(best_frame is None):
                best_frame = frame

            # Create the directory if it does not exist 
            os.makedirs(images_dir, exist_ok=True)

            # Save the screenshot to the specified file 
            full_path = os.path.join(images_dir, filename + '.png')
            cv2.imwrite(full_path, frame)
            print("Get static frame, Saved to path:", full_path)
                
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = time.time()
        self.frame_buffer.queue.clear()

        capture_thread = threading.Thread(target=self.__get_window_frame, args=(self.frame_buffer,))
        process_thread = threading.Thread(target=__detect_static_frame, args=(self,self.frame_buffer))

        capture_thread.start()
        process_thread.start()

        capture_thread.join()
        process_thread.join()

        print("round over")
        return()