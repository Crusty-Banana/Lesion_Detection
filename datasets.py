import os
import pandas as pd
import pydicom
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision import datasets
import numpy as np

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir):
        """
        Args:
            csv_path (str): Path to the CSV file containing annotations.
            image_dir (str): Directory containing the DICOM images.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transforms = transforms.Compose([
            transforms.Resize((1024, 1024)), 
        ])
        self.label_map = {
            "Osteophytes": 1,
            "Disc space narrowing": 2,
            "Surgical implant": 3,
            "Foraminal stenosis": 4,
            "Spondylolysthesis": 5,
            "Vertebral collapse": 6,
            "Other lesions": 7,
            "No finding": 8,
        }
        self.class_map = {
            "Osteophytes": "Abnormal",
            "Disc space narrowing": "Abnormal",
            "Surgical implant": "Abnormal",
            "Foraminal stenosis": "Abnormal",
            "Spondylolysthesis": "Abnormal",
            "Vertebral collapse": "Abnormal",
            "Other lesions": "Abnormal",
            "No finding": None,
        }
        grouped = self.data.groupby('image_id')
        self.data_by_image = [
            {
                "image_id": image_id,
                "boxes": grouped.get_group(image_id)[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist(),
                "labels": grouped.get_group(image_id)['lesion_type'].map(self.label_map).values.tolist(),
                "classes": grouped.get_group(image_id)['lesion_type'].map(self.class_map).values.tolist()
            }
            for image_id in grouped.groups
        ]
        
    def __len__(self):
        return len(self.data_by_image)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with `images` (torch.Tensor) and `targets` (dict with keys `boxes` and `labels`).
        """
        record = self.data_by_image[idx]

        # Load DICOM image
        image_path = os.path.join(self.image_dir, f"{record['image_id']}.dicom")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"DICOM file not found: {image_path}")

        dicom_image = pydicom.dcmread(image_path).pixel_array
        image = torch.tensor(dicom_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Normalize to [0, 1] and convert to 3-channel format
        image = F.convert_image_dtype(image, dtype=torch.float32)
        image = image.expand(3, -1, -1)  # Convert to 3 channels (RGB-like)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        image = (image - mean) / std  # Normalize for pretrained model
        # image = self.transforms(image)

        boxes = torch.tensor(record["boxes"], dtype=torch.float32)
        labels = torch.tensor(record["labels"], dtype=torch.int64)

        # Prepare target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        # Try to resolve no finding sample in training but no luck.
        if (labels[0] == 8):
            target = {
                "boxes": torch.as_tensor(np.array(np.zeros((0, 4)), dtype=float)),
                "labels": torch.as_tensor(np.array([], dtype=int), dtype=torch.int64)
            } 
        return image, target

class LesionDetectionDataset(LesionDataset):
    def __init__(self, csv_path, image_dir):
        """
        Args:
            csv_path (str): Path to the CSV file containing annotations.
            image_dir (str): Directory containing the DICOM images.
        """
        super().__init__(csv_path, image_dir)
        # Uncomment if want to remove negative sample
        # self.data_by_image = [
        #     data
        #     for data in self.data_by_image
        #     if 8 not in data["labels"]
        # ]