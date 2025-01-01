import os
import pandas as pd
import pydicom
import torch
from torchvision.transforms import functional as F

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transforms = transforms
        self.label_map = {
            "Osteophytes": 1,
            "Disc space narrowing": 2,
            "Surgical implant": 3,
            "Foraminal stenosis": 4,
            "Spondylolysthesis": 5,
            "Vertebral collapse": 6,
            "Other lesions": 7,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        
        # Load DICOM image
        image_path = os.path.join(self.image_dir, f"{record['image_id']}.dicom")
        dicom_image = pydicom.dcmread(image_path).pixel_array
        image = torch.tensor(dicom_image).unsqueeze(0)  # Add channel dimension

        # Normalize image to [0, 1] 
        # Convert to RGB (3 channels)
        image = F.convert_image_dtype(image, dtype=torch.float32)
        image = image.expand(3, -1, -1)

        # Target: boxes and labels
        xmin, ymin, xmax, ymax = record["xmin"], record["ymin"], record["xmax"], record["ymax"]
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([self.label_map[record["lesion_type"]]], dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
