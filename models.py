import torch
from tqdm import tqdm  # Import tqdm
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from typing import Dict, List, Tuple
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
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



   
class CustomDenseNet(_nn.Module):
    def __init__(self, pretrained, input_channels, depth, out_features,):
        super().__init__()
        
        assert input_channels == 3
        assert depth in [121,161,169,201]
        
        for name in out_features:
            assert name in ["dense1", "dense2", "dense3", "dense4"]
        self._out_features = out_features
        self._out_feature_strides = {"dense1": 8, "dense2": 16, "dense3": 32, "dense4": 32}
        self._out_feature_channels = {}
        
        if depth == 121:
            _densenet = torchvision.models.densenet121(pretrained=pretrained)
        elif depth == 161:
            _densenet = torchvision.models.densenet161(pretrained=pretrained)
        elif depth == 169:
            _densenet = torchvision.models.densenet169(pretrained=pretrained)
        elif depth == 201:
            _densenet = torchvision.models.densenet201(pretrained=pretrained)
        self.stem = nn.Sequential(
            _densenet.features.conv0,
            _densenet.features.norm0,
            _densenet.features.relu0,
            _densenet.features.pool0,
        )

        self.block1 = _densenet.features.denseblock1
        self.transition1 = _densenet.features.transition1
        self._out_feature_channels["dense1"] = self.transition1.conv.out_channels

        self.block2 = _densenet.features.denseblock2
        self.transition2 = _densenet.features.transition2
        self._out_feature_channels["dense2"] = self.transition2.conv.out_channels

        self.block3 = _densenet.features.denseblock3
        self.transition3 = _densenet.features.transition3
        self._out_feature_channels["dense3"] = self.transition3.conv.out_channels

        self.block4 = _densenet.features.denseblock4
        self.norm5 = _densenet.features.norm5
        self._out_feature_channels["dense4"] = _densenet.classifier.in_features

        del _densenet

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        x = self.block1(x)
        x = self.transition1(x)
        if "dense1" in self._out_features:
            outputs["dense1"] = x
        
        x = self.block2(x)
        x = self.transition2(x)
        if "dense2" in self._out_features:
            outputs["dense2"] = x
        
        x = self.block3(x)
        x = self.transition3(x)
        if "dense3" in self._out_features:
            outputs["dense3"] = x
        
        x = self.block4(x)
        x = self.norm5(x)
        if "dense4" in self._out_features:
            outputs["dense4"] = x
        
        return outputs
    def output_shape(self):
        return {name:self._out_feature_channels[name] for name in self._out_features}
        
class SimpleHead(nn.Module):
    """
    
    """
    def __init__(self, input_shapes, n_classes=1):
        super().__init__()
        self.input_shapes = input_shapes
        heads = []
        for input_shape in input_shapes:
            heads.append(nn.Linear(input_shape, n_classes))
        
        self.heads = nn.ModuleList(heads)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, features, image_sizes = None):
        """
        features: list of feature maps
        image_sizes: list of (H,W)

        return list of logits of shape (N,M) where M is number of classes
        """
        logits = []
        for feature, head, shape in zip(features, self.heads, self.input_shapes):
            pooled_feature = self.flatten(self.avg_pool(feature))
            logits.append(head(pooled_feature))

        return logits
             
class Classifier(nn.Module):
    def __init__(
        self,
        device,
        pixel_mean = [103.530, 116.280, 123.675],
        pixel_std = [1.,1.,1.],
        classes = ["Abnormal"],
        in_features = ["dense4"],
        backbone_depth = 121,
        backbone_out_features = ["dense4"],
    ):
        """
        Initializes the Classifier.

        Args:
            device (torch.device): The device for computation.
            pixel_mean (list): Mean values for input normalization.
            pixel_std (list): Std values for input normalization.
            classes (int): Classes for classification.
            backbone_depth (int): Depth of DenseNet backbone (121, 161, 169, 201).
            backbone_out_features (list): Features to extract from DenseNet backbone.
        """
        super().__init__()
        self.device = device

        # Normalize input images
        self.pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.head_in_features = in_features
        self.size_divisibility = 32

        # Backbone initialization
        self.backbone = CustomDenseNet(
            pretrained=True,
            input_channels = len(pixel_mean),
            depth=backbone_depth,
            out_features=backbone_out_features,
        )
        
        
        backbone_shape = self.backbone.output_shape()
        head_input_shapes = [backbone_shape[f] for f in in_features]
        # Classifier head initialization
        self.head = SimpleHead(
            input_shapes=head_input_shapes,
            n_classes=len(classes),
        )
        
    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        batched_inputs: N-length list of data_dict with following items:
            "image": image tensor of shape (C,H,W)
            "classes": binary tensor of shape (M,) where M is number of classes # for training only
        
        return 
            if eval mode:
                prob tensor of shape (N,M)
            if training:
                dict of losses:
                    "<<loss_name>>": loss value
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        features = [features[f] for f in self.head_in_features]
        logits = self.head(features)
        
        if self.training:
            targets = torch.stack([x["classes"].to(self.device) for x in batched_inputs])
            return self.loss_fn(logits, targets)
        
        else:
            return self.inference(logits)
    
    def preprocess_image(
        self,
        batched_inputs: Tuple[Dict[str, Tensor]],
    ):
        """
        Normalize, pad, and batch the input images.

        Args:
            batched_inputs (Tuple[Dict[str, Tensor]]): A batch of input dictionaries containing images.
            size_divisibility (int): Ensures height and width are divisible by this value.

        Returns:
            padded_images (torch.Tensor): A batch of padded and normalized images of shape (N, C, H, W).
            original_sizes (List[Tuple[int, int]]): Original (H, W) sizes of the images before padding.
        """
        # Move images to the correct device and normalize
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        # Get the original sizes of images
        original_sizes = [img.shape[1:] for img in images]  # (H, W)

        # Calculate max dimensions
        max_height = max(size[0] for size in original_sizes)
        max_width = max(size[1] for size in original_sizes)

        # Adjust dimensions to be divisible by size_divisibility
        if self.size_divisibility > 1:
            max_height = (max_height + self.size_divisibility - 1) // self.size_divisibility * self.size_divisibility
            max_width = (max_width + self.size_divisibility - 1) // self.size_divisibility * self.size_divisibility

        # Pad each image to the maximum size
        padded_images = []
        for img in images:
            padding = [
                0, max_width - img.shape[2],  # Width padding (right)
                0, max_height - img.shape[1],  # Height padding (bottom)
            ]
            padded_images.append(torch.nn.functional.pad(img, padding, value=0))  # Zero padding

        # Stack the images into a single batch tensor
        padded_images = torch.stack(padded_images)

        return padded_images

        
    def inference(self, logits):
        """
        logits: dict["<<feature_map/logit name>>": tensor of shape (N,M)]
        
        return prob of shape (N,M)
        """
        
        stacked_logits = torch.stack(logits)
        scores = stacked_logits.mean(dim=0).sigmoid_()
        return scores
    
    
