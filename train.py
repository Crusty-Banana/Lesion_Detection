from torch.utils.data import DataLoader
from models import CustomLesionDetector, get_custom_faster_rcnn_model, get_custom_retina_net_model
from datasets import LesionDataset, LesionDetectionDataset

def train_model_with_dataset(data_path="", 
                             image_dir="",
                             model_type="FasterRCNN",
                             model_path="",
                             batch_size=4, num_workers=4, device="cpu",
                             checkpoint_path="models/new_model",
                             device_ids=[],
                             epochs=10, learning_rate=0.001):
    """Train a model using a dataset.

    Args:
        data_path (string): Path to csv data.
        image_dir (string): Path to image directory.
        model_path (string): Path to load the model.
        batch_size (string): Bruh.
        device (string): Bruh.
        checkpoint_path (string): Path to save the model.
    """
    dataset = LesionDetectionDataset(data_path, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
    
    model = None
    if model_type == "FasterRCNN":
        model = CustomLesionDetector(model=get_custom_faster_rcnn_model(),device=device, device_ids=device_ids)
    elif (model_type == "RetinaNet"):
        model = CustomLesionDetector(model=get_custom_retina_net_model(), device=device, device_ids=device_ids)

    if (model_path != ""):
        model.load_model(model_path)

    model.train(train_dataloader=dataloader, epochs=epochs, learning_rate=learning_rate, checkpoint_path=checkpoint_path)

    model.save_model(checkpoint_path)

def evaluate_model_with_dataset(data_path="", 
                                image_dir="",
                                model_path="",
                                batch_size=4, device="cpu"):
    """Evaluate a model using a dataset.

    Args:
        data_path (string): Path to csv data.
        image_dir (string): Path to image directory.
        model_path (string): Path to load the model.
        batch_size (string): Bruh.
        device (string): Bruh.
    """

    dataset = LesionDataset(data_path, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

    model = CustomLesionDetector(model=get_custom_faster_rcnn_model(), device=device)
    if (model_path != ""):
        model.load_model(model_path)

    model.evaluate(dataloader)
