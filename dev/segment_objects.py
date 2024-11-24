import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Function to show annotations on the original image
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Function to eliminate non-contiguous mask regions and keep the largest contiguous part
def keep_largest_contiguous_mask(mask):
    # Convert the mask to uint8 format for contour detection (binary image with 0 and 255)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return an empty mask
    if len(contours) == 0:
        return np.zeros_like(mask), None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty uint8 mask (binary image with 0 and 255)
    largest_mask = np.zeros_like(mask_uint8)
    
    # Convert single-channel mask to 3-channel image for drawing
    largest_mask_3ch = cv2.cvtColor(largest_mask, cv2.COLOR_GRAY2BGR)
    
    # Draw the largest contour on the mask
    cv2.drawContours(largest_mask_3ch, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Convert the filled contour mask back to a single-channel binary mask
    largest_mask_final = cv2.cvtColor(largest_mask_3ch, cv2.COLOR_BGR2GRAY)
    
    # Convert back to boolean format (True for mask area, False for background)
    return largest_mask_final.astype(bool), largest_contour

# Function to crop the image and mask tightly around the largest contour
def crop_to_contour(image, mask, contour):
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the image and mask using the bounding box
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Apply the mask to the cropped image
    cropped_image_with_mask = cropped_image.copy()
    cropped_image_with_mask[~cropped_mask] = 0  # Set background to black
    
    return cropped_image_with_mask

# Function to crop and save the segmented objects based on mask's bounding box (bbox)
def save_cropped_objects(anns, image, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        # delete all files in the folder
        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))

    for i, ann in enumerate(anns):
        # Get the bounding box (bbox) for each mask
        x, y, w, h = ann["bbox"]

        # Crop the image using the bounding box
        cropped_image = image[int(y) : int(y + h), int(x) : int(x + w)]
        cropped_mask = ann['segmentation'][int(y):int(y+h), int(x):int(x+w)]
        
        # Keep only the largest contiguous part of the mask
        largest_contiguous_mask, largest_contour = keep_largest_contiguous_mask(cropped_mask)
        
        # Crop the image and mask tightly to the largest contour
        cropped_image_with_mask = crop_to_contour(cropped_image, largest_contiguous_mask, largest_contour)
        
        save_path = os.path.join(save_folder, f'cropped_object_{i+1}_largest_contour.png')
        cv2.imwrite(save_path, cv2.cvtColor(cropped_image_with_mask, cv2.COLOR_RGB2BGR))


# Read and preprocess the image
image = cv2.imread("./screenshots/dragon/1.png")
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    # points_per_batch=64,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.9,
    stability_score_offset=0.7,
    crop_n_layers=0,
    # box_nms_thresh=0.7,
    # crop_n_points_downscale_factor=2,
    min_mask_region_area=600.0,
    # use_m2m=True,
)

# Generate masks
start_time = time.time()
masks = mask_generator.generate(image)
end_time = time.time()

print(f"Elapsed time: {end_time - start_time}")
print(f"len(maks): {len(masks)}")

# Display the original image with annotations
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()

# Crop and save segmented objects
save_cropped_objects(masks, image, save_folder="cropped_objects")
