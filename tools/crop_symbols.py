import cv2
import sys
from pathlib import Path
import numpy as np
import os
from rembg import remove
from PIL import Image

root_dir = Path(__file__).parent.parent
print(root_dir)

GAME = 'Plusz_Joker_sJewels'
MODE = 'base'
SCALE = 0.5

image_path = root_dir / 'images' / f'{GAME}' / 'info' / '2.png'
image = cv2.imread(str(image_path))

# Create the output directory if it doesn't exist
output_dir = root_dir / 'images' / f'{GAME}' / 'symbols'
output_dir.mkdir(parents=True, exist_ok=True)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found.")
    sys.exit(1)
else:
    print(f"Image loaded successfully with shape: {image.shape}")

# Scale the image
scaled_image = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

# Display the scaled image and select multiple ROIs
rois = cv2.selectROIs('Select Symbols', scaled_image, showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()
print(f"Number of ROIs selected: {len(rois)}")

cropped_images = []

# Adjust the ROI coordinates back to the original image scale
for i, rect in enumerate(rois):
    x, y, w, h = rect
    # Rescale the ROI coordinates to match the original image size
    x = int(x / SCALE)
    y = int(y / SCALE)
    w = int(w / SCALE)
    h = int(h / SCALE)
    
    cropped = image[y:y+h, x:x+w]
    cropped_images.append(cropped)

border = 0
processed_images = []
for i, cropped in enumerate(cropped_images):
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
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
    
    # Create a blank mask and draw the largest contour
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Erode the mask to shrink the contour
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    
    x, y, w, h = cv2.boundingRect(eroded_mask)
    cropped_output = output_np[y+border:y+h-border, x+border:x+w-border]
    cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGRA)
    
    # Apply thresholding to the alpha channel
    alpha_channel = cropped_output[:, :, 3]
    _, binary_alpha = cv2.threshold(alpha_channel, 16, 255, cv2.THRESH_BINARY)
    cropped_output[:, :, 3] = binary_alpha

    # Append the tightly cropped image
    processed_images.append(cropped_output)

for i, output_np in enumerate(processed_images):
    # Save the image with transparency (supports PNG format)
    output_path = output_dir / f'symbol_{i}.png'
    cv2.imwrite(str(output_path), output_np)
    print(f"Saved processed image to {output_path}")
