import cv2
import sys
from pathlib import Path
import numpy as np
import os
import math
import time

root_dir = Path(__file__).parent.parent
print(root_dir)

GAME_NAME = 'VIVA FROST VEGAS'
symbol_number = (4,5)
start_frame = 0

SAVE = False
SET_ROI = True
DEBUG = True
bull_mode = False


video_path = root_dir / 'images' / f'{GAME_NAME}' / f'{GAME_NAME}.mkv'
save_path = root_dir / 'images' / f'{GAME_NAME}' / 'screenshots' / 'base_game'

# setting Shi-Tomasi
feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=20, blockSize=20)

# setting Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 10), maxLevel=3, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))

# setting my parameter
rolling_record_size = 9
min_moving_down_distance = 20
min_point_number = 10
max_error = 35
min_rolling_frames = 15
max_degree = 5

video = cv2.VideoCapture(video_path)
video.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
ret, old_frame = video.read()
max_tan = math.tan(math.radians(max_degree))
is_first = True
point_record = [0]
rolling_record = [False]
rolling_frames = 0
prev_point_number = 0
rolling_flag = False
arrow_flag = False
noice_count = 0
max_noice = 1
last_capture_frame = -999
frame_number = 0
arrow_combo = 0

def point_in_rect(point,rect):
    px, py = point
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h

stop_times = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    if is_first:
        if SET_ROI:
            roi = cv2.selectROIs('Select player display', frame, showCrosshair=False, fromCenter=False)[0]
            cv2.destroyAllWindows()
            roi_x, roi_y, roi_w, roi_h = roi
            sh = int(roi_h/symbol_number[0])
            sw = int(roi_w/symbol_number[1])
            lk_params = dict(winSize=(int(sh/7), int(sw/10)), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))
            min_moving_down_distance = sh/5
            min_point_number = min(int(roi_w*roi_h/100000+1),10)
        else:
            roi_x, roi_y, roi_w, roi_h = (300,200,1320,700)
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
    p0 = cv2.goodFeaturesToTrack(new_frame, mask=None, **feature_params)
    rolling_now = False
    rolling_point_number = 0
    arrow_rolling_point_number = 0
    arrow_noise_point_number = 0
    rolling_points_column = [0 for i in range(symbol_number[1])]
    if(len(rolling_record)==rolling_record_size):
        rolling_record.pop(0)
        point_record.pop(0)
    if not is_first and p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            horizontal_range = [9999,-1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                if err[i]<max_error:
                    a, b = int(new.ravel()[0]),int(new.ravel()[1])
                    c, d = int(old.ravel()[0]),int(old.ravel()[1])
                    dx, dy = a - c, b - d
                    if dy >= min_moving_down_distance and abs(dx/dy) <= max_tan:
                        # if DEBUG:
                        #     frame = cv2.line(frame, (a+roi_x, b+roi_y), (c+roi_x, d+roi_y), (0, 255, 0), 2)
                        #     frame = cv2.circle(frame, (a+roi_x, b+roi_y), 5, (0, 0, 255), -1)
                        if int(a/(roi_w/symbol_number[1]))==int(c/(roi_w/symbol_number[1])):
                            rolling_points_column[int(a/(roi_w/symbol_number[1]))] += 1
                            rolling_point_number += 1
                        # if rolling_point_number >= min_point_number:
                        #     rolling_now = True
                        #     break
                    if bull_mode == True and 8 <= frame_number-last_capture_frame <= 25:

                        if dx**2+dy**2 > 4 and dx**2+dy**2 < 2912:
                            video_height = frame.shape[0]
                            horizontal_range[0] = min(horizontal_range[0],a,c)
                            horizontal_range[1] = max(horizontal_range[1],a,c)
                            arrow_rolling_point_number += 1
                            if video_height*0.004 < dx**2+dy**2 < video_height*2.8:
                                horizontal_range[0] = min(horizontal_range[0],a,c)
                                horizontal_range[1] = max(horizontal_range[1],a,c)
                                if video_height*0.004 < abs(dy) < video_height*0.02 and abs(dx/dy) <= max_tan:
                                    arrow_rolling_point_number += 1
                            
        if bull_mode == True and 8 <= frame_number-last_capture_frame <= 25:
            points_width = horizontal_range[1]-horizontal_range[0]
            if(DEBUG):
                print(points_width)
            if bull_mode == True and points_width <= 2*sw and points_width > 0 and arrow_rolling_point_number>0:
                arrow_flag = True
                rolling_now = True
            elif points_width > 2*sw:
                noice_count += 1
        else:
            if DEBUG and frame_number-last_capture_frame == 26:
                print(arrow_flag,noice_count,arrow_combo)
            noice_count = 0
            arrow_flag = False

    if rolling_point_number >= min_point_number:
        rolling_now = True

    rolling_record.append(rolling_now)
    point_record.append(rolling_point_number)
    prev_point_number = rolling_point_number
    old_frame = new_frame.copy()
    is_first = False
    waiting_time =1
    if(len(rolling_record)==rolling_record_size):
        if True in rolling_record:
            if ((rolling_record.index(True) == 0 and rolling_record.count(True) == 1) or frame_number-last_capture_frame==25) and (rolling_frames >= min_rolling_frames or (arrow_flag==True and noice_count <= max_noice and arrow_combo<5)):
                if DEBUG:
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
                    waiting_time = 1000
                    cv2.putText(frame,"rolling frames: "+str(rolling_frames),(roi_x,roi_y-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)
                if SAVE:
                    cv2.imwrite(str(save_path)+f"/{GAME_NAME}_frame{video.get(cv2.CAP_PROP_POS_FRAMES)}.png", frame)
                if(arrow_flag):
                    arrow_combo += 1
                else:
                    arrow_combo = 0
                rolling_frames = 0
                rolling_flag = False
                arrow_flag = False
                last_capture_frame = frame_number
                stop_times+=1
                print(stop_times)
            else:
                if frame_number-last_capture_frame>25:
                    # the wheel is still rolling
                    rolling_frames+=1
                else:
                    rolling_frames = 0

        else:
            if DEBUG:
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
            rolling_frames = 0
    if DEBUG:
        cv2.rectangle(frame, (100, 1000-(min_point_number)), (400, 1000), (255, 255, 255),-1)
        for i in range(len(point_record)):
            if i>0:
                frame = cv2.line(frame, (150+(i-1)*20, 1000-point_record[i-1]), (150+i*20, 1000-point_record[i]), (255, 0, 0), 2)
            if rolling_record[i]==True:
                frame = cv2.circle(frame, (150+i*20, 1000-point_record[i]), 5, (0, 255, 0), -1)
            else:
                frame = cv2.circle(frame, (150+i*20, 1000-point_record[i]), 5, (0, 0, 255), -1)

    

    if DEBUG:
        cv2.imshow('Optical Flow', frame)
        if cv2.waitKey(waiting_time) & 0xFF == 27:
            break
    
    frame_number += 1

video.release()
cv2.destroyAllWindows()