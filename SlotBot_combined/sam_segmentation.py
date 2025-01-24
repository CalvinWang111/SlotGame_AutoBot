import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# 設置torch庫的路徑到PATH環境變量中
torch_path = 'C:\\Users\\13514\\anaconda3\\envs\\Py311\\Lib\\site-packages\\torch\\lib'
os.environ['PATH'] = torch_path + ';' + os.environ['PATH']
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import shutil


class SAMSegmentation:
    def __init__(self, Snapshot, images_dir, sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt", model_cfg = "sam2_hiera_l.yaml"):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        self.device = device
        self.maskDict = {}
        self.output_dir = os.path.join(images_dir, "segmented_image")

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

    def clear_directory(self, directory):
        """
        Deletes all files and subdirectories in the specified directory.
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # Check if it's a file or directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and all its contents
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    def show_anns(self, anns, original_image, borders=True):
        if len(anns) == 0:
            return

        # Clear output directory if it exists, or create it if it doesn't
        if os.path.exists(self.output_dir):
            self.clear_directory(self.output_dir)
        else:
            os.makedirs(self.output_dir)

        # Sort annotations by area (largest to smallest)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        # Create an RGBA image for visualization
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0  # Set the alpha channel to 0 (transparent)

        # Counter for mask files
        mask_counter = 0

        for ann in sorted_anns:
            m = ann['segmentation']

            # Ensure segmentation mask is in uint8 format (0 or 1 values)
            m = m.astype(np.uint8)

            color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color mask with transparency
            img[m == 1] = color_mask  # Update img where mask is 1

            if borders:
                # Find contours in the mask
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Process each contour to extract the corresponding image region
                for i, contour in enumerate(contours):
                    # Get the bounding box for the contour (x, y, width, height)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Crop the original image using the bounding box
                    cropped_image = original_image[y:y+h, x:x+w]

                    # Create a blank mask for the contour in the cropped area
                    cropped_mask = np.zeros((h, w), dtype=np.uint8)

                    # Adjust contour points to the cropped mask's coordinate system
                    contour_cropped = contour - [x, y]

                    # Draw the contour on the cropped mask
                    cv2.drawContours(cropped_mask, [contour_cropped], -1, 255, thickness=cv2.FILLED)

                    # Apply the mask to the cropped image
                    cropped_image_masked = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

                    # Convert masked image to have an alpha channel where the mask is applied
                    cropped_image_rgba = cv2.cvtColor(cropped_image_masked, cv2.COLOR_RGB2RGBA)
                    cropped_image_rgba[:, :, 3] = cropped_mask  # Set alpha channel based on mask

                    # Save the cropped and masked image to the output directory
                    image_filename = os.path.join(self.output_dir, f'cropped_image_{mask_counter}_contour_{i}.png')

                    # Store pixel positions in maskDict
                    self.maskDict[str(f'cropped_image_{mask_counter}_contour_{i}.png')] = (x, y, w, h)
                    cv2.imwrite(image_filename, cropped_image_rgba)  # Save the RGBA image
                    print(f"Saved: {image_filename}")

                # Optionally, draw contours on the main image for visualization
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)  # Draw contour borders

            # Increment the mask counter for each annotation
            mask_counter += 1

            # Display the final image with all masks
            ax.imshow(img)
            
            # Convert the RGBA image to an 8-bit unsigned integer format
            final_image_uint8 = (img * 255).astype(np.uint8)
            
            # Convert RGBA to BGR for saving with OpenCV
            final_image_bgr = cv2.cvtColor(final_image_uint8, cv2.COLOR_RGBA2BGRA)
            
            # Save the final image
            image_filename = os.path.join(self.output_dir, "Final.png")
            cv2.imwrite(image_filename, final_image_bgr)  # Save as BGRA
            print(f"Saved final image: {image_filename}")

    def segment_image(self, route):
        """使用 SAM 分割圖片"""
        image = Image.open(route)
        image = np.array(image.convert("RGB"))

        masks = self.mask_generator.generate(image)
        self.show_anns(masks, image)

        return self.maskDict
