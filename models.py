import torch
from tqdm import tqdm  # Import tqdm
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from sklearn.metrics import confusion_matrix, accuracy_score

class CustomFasterRCNN(_nn.Module):
    def __init__(self,
                    num_classes: int=8):
        super().__init__()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.model = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
    
class CustomLesionDetector:

    def __init__(self, model, device: str="cuda:2"):
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
                optimizer.zero_grad()

                images = batch['image'].to(self.device)
                targets  = batch['target'].to(self.device)

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

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
                images = batch['image'].to(self.device)
                targets  = batch['target'].to(self.device)

                predictions = self.model(images)

                all_labels.extend(targets["labels"].cpu().numpy())
                all_predictions.extend(predictions["labels"].cpu().numpy())

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
