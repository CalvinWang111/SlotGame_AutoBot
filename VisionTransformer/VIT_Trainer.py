import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import ViTForImageClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
from PIL import Image
import os

class VisionTransformerTrainer:
    def __init__(self, train_path, val_path, test_path=None, plot_output=True, batch_size=32, num_epochs=15, learning_rate=1e-5, weight_decay = 0.01):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.plot_output = plot_output
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize data loaders, model, optimizer, and loss function
        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", 
                                                               num_labels=len(self.train_loader.dataset.classes)).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_data = datasets.ImageFolder(root=self.train_path, transform=transform)
        val_data = datasets.ImageFolder(root=self.val_path, transform=transform)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        test_loader = None
        if self.test_path:
            test_data = datasets.ImageFolder(root=self.test_path, transform=transform)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_train_loss = 0.0
            correct_train, total_train = 0, 0

            for images, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)
            
            avg_train_loss = running_train_loss / len(self.train_loader)
            train_accuracy = 100 * correct_train / total_train
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validate after each epoch
            avg_val_loss, val_accuracy = self.validate()
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print epoch metrics
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Plot metrics if enabled
        if self.plot_output:
            self.plot_training_metrics()

    def validate(self):
        self.model.eval()
        running_val_loss = 0.0
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_val_loss / len(self.val_loader)
        val_accuracy = 100 * correct_val / total_val
        return avg_val_loss, val_accuracy

    def plot_training_metrics(self):
        plt.figure(figsize=(12, 5))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, self.num_epochs + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs + 1), self.train_accuracies, label='Training Accuracy')
        plt.plot(range(1, self.num_epochs + 1), self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate_test_set(self):
        if not self.test_loader:
            print("No test set provided.")
            return

        self.model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images).logits
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        # Generate and display confusion matrix
        class_names = self.train_loader.dataset.classes
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        # Print classification report
        print("Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))
        print("Test Accuracy:", accuracy_score(true_labels, pred_labels))
