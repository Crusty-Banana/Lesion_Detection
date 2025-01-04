import os
import pandas as pd
import pydicom
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision import datasets


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
            transforms.Resize((256, 256)),  # Resize to 256x256
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with `images` (torch.Tensor) and `targets` (dict with keys `boxes` and `labels`).
        """
        record = self.data.iloc[idx]

        # Load DICOM image
        image_path = os.path.join(self.image_dir, f"{record['image_id']}.dicom")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"DICOM file not found: {image_path}")

        dicom_image = pydicom.dcmread(image_path).pixel_array
        image = torch.tensor(dicom_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Normalize to [0, 1] and convert to 3-channel format
        image = F.convert_image_dtype(image, dtype=torch.float32)
        image = image.expand(3, -1, -1)  # Convert to 3 channels (RGB-like)
        image = self.transforms(image)

        xmin, ymin, xmax, ymax = float(record["xmin"]), float(record["ymin"]), float(record["xmax"]), float(record["ymax"])
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([self.label_map[record["lesion_type"]]], dtype=torch.int64)

        # Prepare target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target

# class AquariumDetection(datasets.VisionDataset):
#     def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
#         # the 3 transform parameters are reuqired for datasets.VisionDataset
#         super().__init__(root, transforms, transform, target_transform)
#         self.split = split #train, valid, test
#         self.ids = list(sorted(self.coco.imgs.keys()))
#         self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
#     def _load_image(self, id: int):
#         path = self.coco.loadImgs(id)[0]['file_name']
#         image = cv2.imread(os.path.join(self.root, self.split, path))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image
#     def _load_target(self, id):
#         return self.coco.loadAnns(self.coco.getAnnIds(id))
    
#     def __getitem__(self, index):
#         id = self.ids[index]
#         image = self._load_image(id)
#         target = self._load_target(id)
#         target = copy.deepcopy(self._load_target(id))
        
#         boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
#         if self.transforms is not None:
#             transformed = self.transforms(image=image, bboxes=boxes)
        
#         image = transformed['image']
#         boxes = transformed['bboxes']
        
#         new_boxes = [] # convert from xywh to xyxy
#         for box in boxes:
#             xmin = box[0]
#             xmax = xmin + box[2]
#             ymin = box[1]
#             ymax = ymin + box[3]
#             new_boxes.append([xmin, ymin, xmax, ymax])
        
#         boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
#         targ = {} # here is our transformed target
#         targ['boxes'] = boxes
#         targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
#         targ['image_id'] = torch.tensor([t['image_id'] for t in target])
#         targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
#         targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
#         return image.div(255), targ # scale images
#     def __len__(self):
#         return len(self.ids)