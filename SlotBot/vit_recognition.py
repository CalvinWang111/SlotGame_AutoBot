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


class ViTRecognition:
    # put your own model path here
    def __init__(self, Snapshot, maskDict, model_path=r'C:\Users\13514\button_recognition\VITrun_ver6\best_model.pth'):
        # Define the label mapping
        label_map = {
            0: "button_max_bet",
            1: "button_additional_bet",
            2: "button_close",
            3: "button_decrease_bet",
            4: "button_home",
            5: "button_increase_bet",
            6: "button_info",
            7: "button_speedup_spin",
            8: "button_start_spin",
            9: "button_three_dot",
            10: "gold_coin",
            11: "gold_ingot",
            12: "stickers"
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
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=13)
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

    def classify_components(self):
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
                        if (pred_class not in highest_confidence_images) or (confidence > highest_confidence_images[pred_class]['confidence']):
                            
                            highest_confidence_images[pred_class] = {'path': img_path, 'confidence': confidence, 'contour': self.maskDict[img_name]}

        # Copy and display only the highest-confidence images for each class
        for class_id, info in highest_confidence_images.items():
            base_name = f"{self.label_map[class_id]}.png"
            dest_path = os.path.join(template_folder, base_name)
            
            # Copy the file to the "template" folder with the class name
            shutil.copy(info['path'], dest_path)

            # Display the copied image with the confidence score
            plt.figure(figsize=(3, 3))
            plt.imshow(Image.open(dest_path).convert("RGB"))
            plt.axis('off')
            plt.title(f"{self.label_map[class_id]} - Confidence: {info['confidence']:.2f}")
            plt.show()
            print(f"position:{info['contour'][0],info['contour'][1]}, contour Height and Width:{info['contour'][2],info['contour'][3]}")

        return highest_confidence_images