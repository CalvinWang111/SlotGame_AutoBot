import math
import time
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
import os
from rembg import remove
from PIL import Image
import io

root_dir = Path(__file__).parent.parent
print(root_dir)


GAME = 'Plusz_CoinStrikeHoldAndWin'
MODE = 'base'
SCALE = 1

#image_path = root_dir / 'images' / f'{GAME}' / 'info' / 'info (0).png'
image_path = root_dir / 'images' / f'{GAME}' / 'screenshots' / 'base_game' / 'Plusz_CoinStrikeHoldAndWin_frame6605.0.png'
image = cv2.imread(str(image_path))

# Create the output directory if it doesn't exist
output_dir = root_dir / 'images' / f'{GAME}' / 'symbols' / 'base_game'
output_dir.mkdir(parents=True, exist_ok=True)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found.")
    sys.exit(1)
else:
    print(f"Image loaded successfully with shape: {image.shape}")
    
# Display the image and select multiple ROIs
rois = cv2.selectROIs('Select Symbols', image, showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()
print(f"Number of ROIs selected: {len(rois)}")
print(rois)
cropped_images = []

def color_dis(color1,color2):
    if(type(color1)==np.ndarray):
        color1 = color1.astype(np.int32)
    if(type(color2)==np.ndarray):
        color2 = color2.astype(np.int32)
    return math.sqrt((color1[0]-color2[0])**2+(color1[1]-color2[1])**2+(color1[2]-color2[2])**2)

def first_nonzero(arr):
    indices = np.where(arr != 0)
    first_position = (indices[0][0], indices[1][0]) if indices[0].size > 0 else None
    if(first_position!=None):
        return(arr[first_position])
    else:
        return -1
    
def first_zero(arr):
    indices = np.where(arr == 0)
    first_position = (indices[0][0], indices[1][0]) if indices[0].size > 0 else None
    if(first_position!=None):
        return(0)
    else:
        return -1

for i, rect in enumerate(rois):
    x, y, w, h = rect
    padding = 100
    cropped = np.zeros([h+padding*2, w+padding*2, image.shape[2]],np.uint8)
    cropped[padding:padding+h,padding:padding+w] = image[y:y+h, x:x+w]

    avg_color_t = np.mean(image[y, x:x+w], axis=0)
    avg_color_b = np.mean(image[y+h, x:x+w], axis=0)
    avg_color_l = np.mean(image[y:y+h, x], axis=0)
    avg_color_r = np.mean(image[y:y+h, x+w], axis=0)
    avg_avg_color = (avg_color_t+avg_color_b+avg_color_l+avg_color_r)/4
    max_diff = 50
    #print(color_dis(avg_color_t,avg_avg_color),color_dis(avg_color_b,avg_avg_color),color_dis(avg_color_l,avg_avg_color),color_dis(avg_color_r,avg_avg_color))
    if(color_dis(avg_color_t,image[y, x])>max_diff and color_dis(avg_color_t,image[y, x+w]>max_diff)):
        avg_color_t = [0,0,0]
    if(color_dis(avg_color_b,image[y+h, x])>max_diff and color_dis(avg_color_b,image[y+h, x+w]>max_diff)):
        avg_color_b = [0,0,0]
    if(color_dis(avg_color_l,image[y, x])>max_diff and color_dis(avg_color_l,image[y+h, x]>max_diff)):
        avg_color_l = [0,0,0]
    if(color_dis(avg_color_r,image[y, x+w])>max_diff and color_dis(avg_color_r,image[y+h, x+w]>max_diff)):
        avg_color_r = [0,0,0]

    cropped[:padding,:padding+w] = avg_color_t
    cropped[padding+h:,padding:] = avg_color_b
    cropped[padding:,:padding] = avg_color_l
    cropped[:padding+h,padding+w:] = avg_color_r
    
    # cv2.imshow("test", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cropped_images.append(cropped)

processed_images = []
for i, cropped in enumerate(cropped_images):   
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # mark some big and monotonous zone as background
    class background_mask:
        def __init__(self):
            self.area = 0
            self.mask = any
            self.color = [255,255,255]
            self.rect = (0,0,0,0)
    
    monochrome_zones = []
    marked = np.zeros((cropped_rgb.shape[0]+2, cropped_rgb.shape[1]+2)) 
    marked = marked.astype(np.uint8)

    for i in range(cropped_rgb.shape[0]):
        for j in range(cropped_rgb.shape[1]):
            if marked[i][j]==0:
                original_mark = marked.copy()
                tolerance = 10
                mask = background_mask()
                # mask.color = (cropped_rgb[i,j,0],cropped_rgb[i,j,1],cropped_rgb[i,j,2])
                mask.area, _, mask.mask, mask.rect = cv2.floodFill(cropped_rgb, marked, [j,i], (0,0,0), (tolerance,tolerance,tolerance), (tolerance,tolerance,tolerance), cv2.FLOODFILL_MASK_ONLY)
                if(mask.area >=30 and mask.rect[2]>1 and mask.rect[3]>1):
                    mask.mask = marked - original_mark
                    masked_colors = cropped_rgb[mask.mask[1:-1,1:-1]==1]
                    avg_color = masked_colors.mean(axis=0)
                    mask.color = avg_color
                    monochrome_zones.append(mask)

    cropped_pil = Image.fromarray(cropped_rgb)

    # Remove background
    output = remove(cropped_pil)
    output_np = np.array(output)

    alpha_channel = output_np[:, :, 3]
    _, thresh = cv2.threshold(alpha_channel, 32, 255, cv2.THRESH_BINARY)
    
    # Dilate to merge nearby contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the monotonous zones that might be the background
    largest_contour_np = np.zeros_like(alpha_channel)
    cv2.drawContours(largest_contour_np, [largest_contour], -1, 255, thickness=1)
    bg_zones = []
    bg_color = []

    test = cropped.copy()
    cv2.drawContours(test, [largest_contour], -1, 255, thickness=1)
    for point in largest_contour:
        bg_color.append(cropped_rgb[point[0,1],point[0,0],0:3])
    
    # Create a blank mask and draw the largest contour
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Erode the mask to shrink the contour
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    
    for zone in bg_zones:
        eroded_mask = cv2.bitwise_and(eroded_mask, ((1-zone.mask[1:-1,1:-1])*255).astype(np.uint8))

    # filter small hole and fragments again
    _, thresh = cv2.threshold(eroded_mask, 32, 255, cv2.THRESH_BINARY)
    
    # Dilate to merge nearby contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask and draw the largest contour
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Erode the mask to shrink the contour
    eroded_mask = cv2.erode(mask, kernel, iterations=2)

    
    x, y, w, h = cv2.boundingRect(eroded_mask)
    cropped_output = output_np[y:y+h, x:x+w]
    cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGRA)
    cropped_mask = eroded_mask[y:y+h, x:x+w]
    
    # Apply thresholding to the alpha channel
    binary_alpha = cropped_mask
    cropped_output[:, :, 3] = binary_alpha

    # Append the tightly cropped image
    processed_images.append(cropped_output)

for i, output_np in enumerate(processed_images):
    # Save the image with transparency (supports PNG format)
    output_path = output_dir / f'symbol_{time.time()}.png'
    cv2.imwrite(str(output_path), output_np)
    print(f"Saved processed image to {output_path}")
    # cv2.imshow("test",output_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()