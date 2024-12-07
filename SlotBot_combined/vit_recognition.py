from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil  # For copying files
import torch.nn.functional as F  # For softmax
import json


class ViTRecognition:
    # put your own model path here
    def __init__(self, Snapshot, maskDict, model_path=r'C:\Users\13514\button_recognition\VITrun_ver6\best_model.pth'):
        # Define the label mapping
        label_map = {
                    0: "button_max_bet",
                    1: "button_additional_bet",
                    2: "button_close",
                    3: "confirm",
                    4: "button_decrease_bet",
                    5: "button_home",
                    6: "button_increase_bet",
                    7: "button_info",
                    8: "receive",
                    9: "button_speedup_spin",
                    10: "button_start_spin",
                    11: "button_three_dot",
                    12: "gold_coin",
                    13: "gold_ingot",
                    14: "stickers",
                }

        # Confidence threshold (e.g., 0.8 for 80%)
        confidence_threshold = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for ViT
        ])

        # Load the model and move it to the device
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=15)
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        self.test_folder = "./" + Snapshot
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.true_labels = []
        self.pred_labels = []
        self.image_cor = []
        self.maskDict = maskDict
        self.transform = transform
        self.model = model

    def output_json(self, highest_confidence_images, template_folder):
        # Transform the dictionary keys using the mapping
        transformed_dict = {}
        for key, value in highest_confidence_images.items():
            new_key = self.label_map.get(key, str(key))  # Default to string key if no mapping
            transformed_dict[new_key] = value

        # Define the output file path
        output_file = os.path.join(template_folder, "_controlcompoment.json")

        # Check if the file already exists and read its content
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as file:
                    existing_data = json.load(file)
            except Exception as e:
                print(f"Error reading existing JSON file: {e}")
                existing_data = {}
        else:
            existing_data = {}

        # Merge the existing data with the new transformed data
        merged_data = {**existing_data, **transformed_dict}  # Union dictionaries (new data overwrites old for same keys)

        # Write the merged data to the JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(merged_data, file, indent=4, ensure_ascii=False)
            print(f"Regions successfully saved to {output_file}")
        except Exception as e:
            print(f"An error occurred while writing to file: {e}")


    def classify_components(self, freegame_compoment=[3, 8, 12, 13 ]):
        """使用 ViT 模型對分割元件進行辨識"""
        # Create the "template" folder if it doesn't exist
        template_folder = os.path.join(self.test_folder, "template")
        os.makedirs(template_folder, exist_ok=True)

        # Dictionary to store the highest-confidence image path for each class
        highest_confidence_images = {}

        # Process each image in the test dataset
        for root, dirs, files in os.walk(self.test_folder):
            for img_name in files:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, img_name)
                    
                    # Load and transform the image
                    image = Image.open(img_path).convert("RGB")
                    image = self.transform(image).unsqueeze(0).to(self.device)

                    # Predict the class
                    with torch.no_grad():
                        outputs = self.model(image).logits
                        probabilities = F.softmax(outputs, dim=1)
                        pred_class = probabilities.argmax().item()
                        confidence = probabilities[0, pred_class].item()  # Confidence of the predicted class

                    # Check if the prediction meets the threshold and is not in class 10
                    if pred_class != 12 and confidence >= self.confidence_threshold:
                        # If the class is not yet in the dictionary or the confidence is higher, update the entry
                        if pred_class in freegame_compoment:
                            # For classes 10 and 11, append all entries
                            if pred_class not in highest_confidence_images:
                                highest_confidence_images[pred_class] = []
                                
                            highest_confidence_images[pred_class].append({
                                'path':img_path, 
                                'confidence':confidence, 
                                'contour':self.maskDict[img_name],
                                'value':None
                            })
                            
                        elif (pred_class not in highest_confidence_images) or (confidence > highest_confidence_images[pred_class]['confidence']):
                            
                            highest_confidence_images[pred_class] = {'path': img_path, 'confidence': confidence, 'contour': self.maskDict[img_name], 'value': None}

        # Copy and display only the highest-confidence images for each class
        for class_id, info in highest_confidence_images.items():
            if class_id not in freegame_compoment:
                base_name = f"{self.label_map[class_id]}.png"
                dest_path = os.path.join(template_folder, base_name)
                
                # Copy the file to the "template" folder with the class name
                shutil.copy(info['path'], dest_path)

                # Display the copied image with the confidence score
                '''
                plt.figure(figsize=(3, 3))
                plt.imshow(Image.open(dest_path).convert("RGB"))
                plt.axis('off')
                plt.title(f"{self.label_map[class_id]} - Confidence: {info['confidence']:.2f}")
                plt.show()
                print(f"position:{info['contour'][0],info['contour'][1]}, contour Height and Width:{info['contour'][2],info['contour'][3]}")
                '''
        return highest_confidence_images, template_folder