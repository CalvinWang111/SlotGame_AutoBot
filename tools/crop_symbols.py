import os.path
import math
import cv2
import sys
from pathlib import Path
import numpy as np
from rembg import remove
from PIL import Image
import json

root_dir = Path(__file__).parent.parent
print(root_dir)

GAME = 'golden'
MODE = 'base'

# 僅影響框選roi時的預覽圖片大小，方便解析度較低的螢幕使用
SCALE = 0.5
# 若設為True，會讀取已存在的json並增加/修改其內容；若設為False，則會完全覆蓋已存在的json檔案
APPEND_JSON = True
# 若設為True，背景移除程式會更容易保留符號自帶的背景或邊框
KEEP_SYMBOL_BORDER = True
# 若設為True，將跳過自動移除背景步驟，直接儲存選取roi內的所有內容，可以在symbol幾乎為方形，且難以正確去背時開啟
SKIP_REMBG = False

image_path = root_dir / 'images' / f'{GAME}' / 'info' / '1.png'

# Create the output directory if it doesn't exist
image_output_dir = root_dir / 'images' / f'{GAME}' / 'symbols' / f'{MODE}_game'
image_output_dir.mkdir(parents=True, exist_ok=True)
json_output_dir = image_output_dir.parent

print("json_output_dir",json_output_dir)

image = cv2.imread(str(image_path))

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found.")
    sys.exit(1)
else:
    print(f"Image loaded successfully with shape: {image.shape}")

# Scale the image
scaled_image = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

# Display the image and select multiple ROIs
rois = cv2.selectROIs('Select Symbols', scaled_image, showCrosshair=False, fromCenter=False)
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

output_bboxes = []
for i, rect in enumerate(rois):
    x, y, w, h = rect
    # Rescale the ROI coordinates to match the original image size
    x = int(x / SCALE)
    y = int(y / SCALE)
    w = int(w / SCALE)
    h = int(h / SCALE)

    if(KEEP_SYMBOL_BORDER and not SKIP_REMBG):
        padding = int(100 / SCALE)
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
        
        output_bboxes.append([x-padding,y-padding,w+padding*2,h+padding*2])
    else:
        cropped = image[y:y+h, x:x+w]
        output_bboxes.append([x,y,w,h])
    
    cropped_images.append(cropped)

border = 0
processed_images = []

for i, cropped in enumerate(cropped_images):
    print(cropped.shape)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    # Remove background
    if not SKIP_REMBG:
        output = remove(cropped_pil)
        output_np = np.array(output)
    
        alpha_channel = output_np[:, :, 3]
        _, thresh = cv2.threshold(alpha_channel, 32, 255, cv2.THRESH_BINARY)
        
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
        x2, y2 = output_bboxes[i][0], output_bboxes[i][1]
        output_bboxes[i] = [x2+x+border,y2+y+border,w-border*2,h-border*2]
        cropped_output = output_np[y+border:y+h-border, x+border:x+w-border]
        cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGRA)
        
        # Apply thresholding to the alpha channel
        alpha_channel = cropped_output[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 32, 255, cv2.THRESH_BINARY)
        cropped_output[:, :, 3] = binary_alpha
    else:
        cropped_output = cropped
    
    # Append the tightly cropped image
    processed_images.append(cropped_output)

json_path = json_output_dir / image_path.name.replace(".png",".json")
if(APPEND_JSON and os.path.exists(json_path)):
    with open(json_path, 'r', encoding='utf-8') as fp:
        json_output = json.load(fp)
else:
    json_output = {
        "info_path": str(image_path.relative_to(root_dir)),
        "info_shape": 
        {
            "width":image.shape[1],
            "height":image.shape[0]
        },
        "symbols":[]
    }

for i, output_np in enumerate(processed_images):
    # Save the image with transparency (supports PNG format)
    cv2.imshow("test",output_np)
    cv2.waitKey(1)
    symbol_name = input("input the name:")
    cv2.destroyAllWindows()

    output_path = image_output_dir / f'{symbol_name}.png'
    cv2.imwrite(str(output_path), output_np)
    print(f"Saved processed image to {output_path}")

    symbol_data = {
        "name":symbol_name,
        "path":str(output_path.relative_to(root_dir)),
        "bbox":output_bboxes[i] # x, y, w, h
    }

    id,counter = -1,0
    for symbol in json_output["symbols"]:
        if symbol["name"] == symbol_name:
            id = counter
        counter += 1
    
    if id != -1:
        json_output["symbols"][id] = symbol_data
    else:
        json_output["symbols"].append(symbol_data)
with open(json_path, 'w', encoding='utf-8') as fp:
    json.dump(obj=json_output,fp=fp,indent=4)