from torch.utils.data import DataLoader
from models import CustomLesionDetector, CustomFasterRCNN
from datasets import LesionDataset

def train_model_with_dataset(data_path="", 
                             image_dir="",
                             model_path="",
                             batch_size=4, device="cpu",
                             checkpoint_path="models/new_model"):
    """Train a model using a dataset.

    Args:
        data_path (string): Path to csv data.
        image_dir (string): Path to image directory.
        model_path (string): Path to load the model.
        batch_size (string): Bruh.
        device (string): Bruh.
        checkpoint_path (string): Path to save the model.
    """
    dataset = LesionDataset(data_path, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = CustomLesionDetector(model=CustomFasterRCNN(), device=device)
    if (model_path != ""):
        model.load_model(model_path)

    model.train(train_dataloader=dataloader, epochs=10, learning_rate=3e-5)

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = CustomLesionDetector(model=CustomFasterRCNN(), device=device)
    if (model_path != ""):
        model.load_model(model_path)

    model.evaluate(dataloader)
