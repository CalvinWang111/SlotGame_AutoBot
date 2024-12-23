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
from grid import BullGrid
from screenshot import GameScreenshot
from game_controller import GameController

target_fps = 30
MAX_BUFFER_SIZE = 32
DEBUG = False
SET_ROI = False         # For testing, will not use data of grid but manually select
if SET_ROI:
    symbol_number = (4,5)

class StoppingFrameCapture:
    def __init__(self,window_name,grid:BullGrid,save_dir, Snapshot, elapsed_time_threshold,game_name):
        self.grid = grid
        self.window_name = window_name
        self.save_dir = save_dir
        self.__output_counter = 0
        self.free_gamestate = False
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = 0      
        self.Snapshot = Snapshot
        self.time_threshold = elapsed_time_threshold
        if(game_name=="bull"):
            self.bull_mode = True
        else:
            self.bull_mode = False
        if DEBUG:
            print("bbox:",grid.bbox)

    def get_key_frames(self, intial_intensity,intensity_threshold,highest_confidence_images):
        key_image_pathes = []

        def __get_window_frame(self:StoppingFrameCapture,frame_buffer):
            window = gw.getWindowsWithTitle(self.window_name)[0]
            left, top, width, height = window.left, window.top, window.width, window.height
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct = mss.mss()
            frame_time = 1/target_fps
            screenshot = GameScreenshot()
            count = 1
            
            while not self.__terminated:
                frame_start_time = time.time() 
                frame = np.array(sct.grab(monitor))

                if frame_buffer.qsize() < MAX_BUFFER_SIZE:
                    frame_buffer.put(frame)
                else:
                    print("Warning: Frame buffer is full")
                
                #avg_intensities = screenshot.clickable(snapshot_path='./images/'+self.Snapshot+'_runtime.png',highest_confidence_images=highest_confidence_images)
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

                frame_elapsed = time.time() - frame_start_time
                if DEBUG:
                    print(f"Frame read time: {frame_elapsed}, Buffer size: {frame_buffer.qsize()}")
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)
                else:
                    print("Warning: Capture speed lower than target frame rate")
                count += 1
            sct.close()
            
        def __detect_stopping_frame(self:StoppingFrameCapture,frame_buffer):
            roi_x, roi_y, roi_w, roi_h = self.grid.bbox
            sh = self.grid.symbol_height
            sw = self.grid.symbol_width
            # adjust detecting area into 3 * 5, whitch can make things easy
            if self.bull_mode:
                roi_h = 3*sh
                if self.grid.growth_direction=="up":
                    roi_y += (self.grid.row - 3)*sh

            # setting Shi-Tomasi
            feature_params = dict(maxCorners=50000, qualityLevel=0.01, minDistance=20, blockSize=20)

            # setting Lucas-Kanade optical flow
            lk_params = dict(winSize=(int(sh/7), int(sw/10)), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


            # setting my parameter
            rolling_record_size = 9
            min_moving_down_distance = sh/5
            min_point_number = min(int(roi_w*roi_h/100000+1),10)
            print("min_point_number:",min_point_number)
            max_error = 25
            min_rolling_frames = 15
            max_degree = 5

            max_tan = math.tan(math.radians(max_degree))
            point_record = [0]
            rolling_record = [False]
            rolling_frames = 0
            is_first = True
            last_capture_time = time.time()
            capture_number = 0
            # some thing about arrow detection
            arrow_rolling_point_number = 0
            arrow_flag = False
            noice_count = 0
            max_noice = 1
            last_capture_frame = -999
            frame_number = 0
            arrow_combo = 0
            elapsed_start_time = time.time()

            while not (self.__terminated==True and frame_buffer.qsize()==0):
                if not frame_buffer.empty():
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
                                if dy >= min_moving_down_distance and abs(dx/dy) <= max_tan:                                    
                                    rolling_point_number += 1
                                    if rolling_point_number >= min_point_number:
                                        rolling_now = True

                                # arrow detection
                                if self.bull_mode == True and 8 <= frame_number-last_capture_frame <= 25:
                                    if noice_count <= max_noice:
                                        video_height = frame.shape[0]
                                        if video_height*0.004 < dx**2+dy**2 < video_height*2.8:
                                            horizontal_range[0] = min(horizontal_range[0],a,c)
                                            horizontal_range[1] = max(horizontal_range[1],a,c)
                                            if video_height*0.004 < abs(dy) < video_height*0.02 and abs(dx/dy) <= max_tan:
                                                arrow_rolling_point_number += 1
                                    
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
                                # the wheel is stop right now, save the image
                                cv2.imwrite(f"{self.save_dir}\\key_frame{self.__output_counter}.png", frame)
                                key_image_pathes.append(f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                if arrow_flag:
                                    print("Get arrow stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                    if(DEBUG):
                                        print(arrow_flag,noice_count,arrow_combo)
                                    arrow_combo += 1
                                else:
                                    print("Get stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                    arrow_combo = 0
                                self.__output_counter+=1
                                last_capture_time = time.time()
                                last_capture_frame = frame_number
                                capture_number += 1
                            else:
                                if frame_number-last_capture_frame>25:
                                    # the wheel is still rolling
                                    rolling_frames+=1
                                else:
                                    rolling_frames = 0
                        else:
                            rolling_frames = 0
                    if DEBUG:
                        print(f"Stopping-judgment time: {time.time()-start_time}, Buffer size: {frame_buffer.qsize()}")
                    
                    frame_number += 1
                
                # prevent busy waiting
                else:
                    time.sleep(0.01) 

                #Free Game state
                elapsed__time = (time.time() - elapsed_start_time)
                #print('elapsed_time', elapsed__time)
                if int(elapsed__time) >= 10 and int(elapsed__time) % 5 == 0 and self.__button_available == False:
                    print('into freegame_control')
                    #print('button state', self.__button_available)
                    GameController.freegame_control(Snapshot=self.Snapshot)
                    self.free_gamestate = True
                elif elapsed__time > 30:
                    self.__terminated = True

                # Termination condition
                if self.__button_available==True:
                    if time.time()-last_capture_time>5 :
                        self.__terminated = True
                
        frame_buffer = Queue()
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = time.time()

        capture_thread = threading.Thread(target=__get_window_frame, args=(self,frame_buffer))
        process_thread = threading.Thread(target=__detect_stopping_frame, args=(self,frame_buffer))

        capture_thread.start()
        process_thread.start()

        capture_thread.join()
        process_thread.join()

        print("round over")
        return(key_image_pathes)