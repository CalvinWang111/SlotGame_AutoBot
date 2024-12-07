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
SAVE_THREAD = False     # enable this if you need faster storage speed
DEBUG = False
SET_ROI = False         # For testing, will not use data of grid but manually select
if SET_ROI:
    symbol_number = (4,5)

class StoppingFrameCapture:
    def __init__(self,window_name,grid:BullGrid,save_dir, Snapshot, elapsed_time_threshold):
        self.grid = grid
        self.window_name = window_name
        self.save_dir = save_dir
        self.__output_counter = 0
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = 0      
        self.Snapshot = Snapshot
        self.time_threshold = elapsed_time_threshold
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
            
            while not self.__terminated:
                frame_start_time = time.time() 
                frame = np.array(sct.grab(monitor))
                if frame_buffer.qsize() < MAX_BUFFER_SIZE:
                    frame_buffer.put(frame)
                else:
                    print("Warning: Frame buffer is full")
                
                '''
                intensity = screenshot.clickable_np(frame,highest_confidence_images=highest_confidence_images)
                if abs(intial_intensity - intensity) < intensity_threshold and time.time()-self.__spin_start_time>1:
                    self.__button_available = True
                else:
                    self.__button_available = False
                '''
                screenshot.capture_screenshot(window_title=self.window_name, filename=self.Snapshot+'_runtime')
                avg_intensities = screenshot.clickable(snapshot_path=self.Snapshot+'_runtime',highest_confidence_images=highest_confidence_images)
                if screenshot.intensity_check(avg_intensities=avg_intensities, intensity_threshold=intensity_threshold):
                    self.__button_available = True
                else:
                    self.__button_available = False

                frame_elapsed = time.time() - frame_start_time
                if DEBUG:
                    print(f"Frame read time: {frame_elapsed}, Buffer size: {frame_buffer.qsize()}")
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)
                else:
                    print("Warning: Capture speed lower than target frame rate")
            sct.close()
            
        def __detect_stopping_frame(self:StoppingFrameCapture,frame_buffer, save_frame_queue):
            roi_x, roi_y, roi_w, roi_h = self.grid.bbox
            sh = self.grid.symbol_height
            sw = self.grid.symbol_width

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

            while not (self.__terminated==True and frame_buffer.qsize()==0):
                if not frame_buffer.empty():
                    start_time = time.time()
                    frame = frame_buffer.get()
                    if is_first:
                        old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
                        is_first = False
                        continue
                    
                    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
                    p0 = cv2.goodFeaturesToTrack(new_frame, mask=None, **feature_params)
                    rolling_now = False
                    rolling_point_number = 0
                    if(len(rolling_record)==rolling_record_size):
                        rolling_record.pop(0)
                        point_record.pop(0)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)
                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            if err[i]<max_error:
                                a, b = int(new.ravel()[0])+roi_x,int(new.ravel()[1])+roi_y
                                c, d = int(old.ravel()[0])+roi_x,int(old.ravel()[1])+roi_y
                                dx, dy = a - c, b - d
                                if dy >= min_moving_down_distance and abs(dx/dy) <= max_tan:                                    
                                    rolling_point_number += 1
                                    if rolling_point_number >= min_point_number:
                                        rolling_now = True
                                        # break
                    rolling_record.append(rolling_now)
                    point_record.append(rolling_point_number)
                    old_frame = new_frame.copy()
                    is_first = False
                    if(len(rolling_record)==rolling_record_size):
                        if True in rolling_record:
                            if rolling_record.index(True) == 0 and rolling_record.count(True) == 1 and rolling_frames >= min_rolling_frames:
                                if not SAVE_THREAD:
                                    cv2.imwrite(f"{self.save_dir}\\key_frame{self.__output_counter}.png", frame)
                                    key_image_pathes.append(f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                    print("Get stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                                    self.__output_counter+=1
                                    last_capture_time = time.time()
                                else:
                                    if save_frame_queue.qsize() < MAX_BUFFER_SIZE:
                                        save_frame_queue.put(frame)
                                    else:
                                        print("Warning: saving queue is full, skipping frame...")
                                capture_number += 1
                            else:
                                rolling_frames+=1
                            
                        else:
                            rolling_frames = 0
                    if DEBUG:
                        print(f"Stopping-judgment time: {time.time()-start_time}, Buffer size: {frame_buffer.qsize()}")
                        
                else:
                    time.sleep(0.01) 

                if self.__button_available==True:
                    if time.time()-last_capture_time>5 :
                        self.__terminated = True

        def save_frame(self:StoppingFrameCapture,save_frame_queue):
            start_time = time.time()

            while not self.__terminated:
                elapsed_time = time.time() - start_time
                if not save_frame_queue.empty():
                    start_time = time.time()
                    cv2.imwrite(f"{self.save_dir}\\key_frame{self.__output_counter}.png", save_frame_queue.get())
                    key_image_pathes.append(f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                    print("Get stopping frame, Saved to path:", f"{self.save_dir}\\key_frame{self.__output_counter}.png")
                    self.__output_counter+=1
                    if DEBUG:
                        print(f"Saving time: {time.time()-start_time}, Queue size: {save_frame_queue.qsize()}")
                elif elapsed_time > self.time_threshold:
                    GameController.freegame_control(window_name=self.window_name, Snapshot=self.Snapshot)
                elif elapsed_time >= self.time_threshold + 20:
                    break
                else:
                    time.sleep(0.01)
                
        frame_buffer = Queue()
        save_frame_queue = Queue()
        self.__button_available = False
        self.__terminated = False
        self.__spin_start_time = time.time()

        capture_thread = threading.Thread(target=__get_window_frame, args=(self,frame_buffer))
        process_thread = threading.Thread(target=__detect_stopping_frame, args=(self,frame_buffer, save_frame_queue))
        if SAVE_THREAD:
            save_thread = threading.Thread(target=save_frame, args=(self,save_frame_queue))   # not necessary for now

        capture_thread.start()
        process_thread.start()
        if SAVE_THREAD:
            save_thread.start()

        capture_thread.join()
        process_thread.join()
        if SAVE_THREAD:
            save_thread.join()

        print("round over")
        return(key_image_pathes)