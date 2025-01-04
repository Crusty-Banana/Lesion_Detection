import torch
from tqdm import tqdm  # Import tqdm
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from sklearn.metrics import confusion_matrix, accuracy_score

def get_custom_faster_rcnn_model(num_classes=9):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
class CustomLesionDetector:

    def __init__(self, model, device: str="cuda"):
        """Initialize the model for lesion detection.

        Args:
            model (class object): Pre-trained Object Detector model.
            device (str): 
        """
        self.model = model
        self.device = device

    def train(self, train_dataloader, epochs=10, learning_rate=1e-3):
        """Train the Lesion Detector.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                images, targets = batch
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.clone().detach().to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

    def evaluate(self, val_dataloader):
        """Evaluate the model on a validation dataset.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
        """
        self.model.to(self.device)
        self.model.eval()
        all_labels = []
        all_predictions = []

        # Wrap the validation dataloader with tqdm to show progress
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                images, targets = batch
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.clone().detach().to(self.device) for k, v in t.items()} for t in targets]

                predictions = self.model(images)
                pred_box, pred_label, pred_score = predictions["boxes"], predictions["labels"], predictions["scores"]
                
                all_labels.extend(targets["labels"].cpu().numpy())
                all_predictions.extend(pred_label.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Validation Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:\n", cm)

    def predict(self, image):
        """Perform inference on a single text.

        Args:
            image: input image.
        """
        self.model.eval()
        self.model.to(self.device)

        prediction = self.model(image)

        return prediction
    
    def save_model(self, save_path):
        """Save the trained model to a file.

        Args:
            save_path (str): Path to save the model.
        """
        torch.save(self.model, save_path+".pth")

        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a model from a file.

        Args:
            load_path (str): Path to load the model from.
        """
        self.model = torch.load(load_path+".pth")
        print(f"Model loaded from {load_path}")
