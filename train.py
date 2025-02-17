from torch.utils.data import DataLoader
from models import (CustomLesionDetector, 
                    CustomAnomalyClassifier,
                    get_custom_faster_rcnn_model, 
                    get_custom_retina_net_model, 
                    get_custom_densenet_121_model,
                    get_custom_densenet_169_model,
                    get_custom_densenet_201_model)
from datasets import LesionDataset, LesionDetectionDataset, LesionClassificationDataset
from helpers import show_object_detection_image, calc_classification_metrics
import torch
from sklearn.metrics import roc_curve
import numpy as np

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
    
    model = None
    if model_type == "FasterRCNN":
        dataset = LesionDetectionDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomLesionDetector(model=get_custom_faster_rcnn_model(),device=device, device_ids=device_ids)
    elif (model_type == "RetinaNet"):
        dataset = LesionDetectionDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomLesionDetector(model=get_custom_retina_net_model(), device=device, device_ids=device_ids)
    elif (model_type == "Densenet121"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_121_model(), device=device, device_ids=device_ids)
    elif (model_type == "Densenet169"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_169_model(), device=device, device_ids=device_ids)
    elif (model_type == "Densenet201"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_201_model(), device=device, device_ids=device_ids)

    if (model_path != ""):
        model.load_model(model_path)

    model.train(train_dataloader=dataloader, epochs=epochs, learning_rate=learning_rate, checkpoint_path=checkpoint_path)

    model.save_model(checkpoint_path)

def evaluate_model_with_dataset(data_path="", 
                                image_dir="",
                                model_path="",
                                model_type="",
                                batch_size=4, num_workers=4, device="cpu"):
    """Evaluate a model using a dataset.

    Args:
        data_path (string): Path to csv data.
        image_dir (string): Path to image directory.
        model_path (string): Path to load the model.
        batch_size (string): Bruh.
        device (string): Bruh.
    """

    model = None
    type = None
    if model_type == "FasterRCNN":
        dataset = LesionDetectionDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomLesionDetector(model=get_custom_faster_rcnn_model(),device=device)
    elif (model_type == "RetinaNet"):
        dataset = LesionDetectionDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomLesionDetector(model=get_custom_retina_net_model(), device=device)
    elif (model_type == "Densenet121"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_121_model(), device=device)
    elif (model_type == "Densenet169"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_169_model(), device=device)
    elif (model_type == "Densenet201"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = CustomAnomalyClassifier(model=get_custom_densenet_201_model(), device=device)
    elif (model_type == "Ensemble"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
        model = [CustomAnomalyClassifier(model=get_custom_densenet_121_model(), device=device),
                CustomAnomalyClassifier(model=get_custom_densenet_169_model(), device=device),
                CustomAnomalyClassifier(model=get_custom_densenet_201_model(), device=device)]
        type = "ensemble"

    if (type == None):
        if (model_path != ""):
            model.load_model(model_path)

        model.evaluate(dataloader)
    elif (type == "ensemble"):
        model[0].load_model("models/densenet_121")
        model[1].load_model("models/densenet_169")
        model[2].load_model("models/densenet_201")

        evaluations = [model[i].evaluate(dataloader) for i in range(3)]
        probs = [eval[0] for eval in evaluations]
        targs = evaluations[0][1]

        mean_probs = [(prob1+prob2+prob3)/3 for prob1, prob2, prob3 in zip(*probs)]

        fpr, tpr, _ = roc_curve(targs,mean_probs)

        c = np.argmax(tpr - fpr)

        preds = [(0 if prob < c else 1) for prob in mean_probs]
        
        print(f"At threshold c = {c:4f}, Youden's J Index is maximized.")
        print("Ensemble of classifier evaluations:")
        calc_classification_metrics(targs, preds, mean_probs)

def inference_model(data_path="", 
                    image_dir="",
                    model_path="",
                    model_type="",
                    device="cuda"):
    """Evaluate a model using a dataset.

    Args:
        data_path (string): Path to csv data.
        image_dir (string): Path to image directory.
        model_path (string): Path to load the model.
        device (string): Bruh.
    """

    model = None
    dataset = None
    type = None
    if model_type == "FasterRCNN":
        dataset = LesionDetectionDataset(data_path, image_dir)
        model = CustomLesionDetector(model=get_custom_faster_rcnn_model(),device=device)
        type = "detection"
    elif (model_type == "RetinaNet"):
        dataset = LesionDetectionDataset(data_path, image_dir)
        model = CustomLesionDetector(model=get_custom_retina_net_model(), device=device)
        type = "detection"
    elif (model_type == "Densenet121"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        model = CustomAnomalyClassifier(model=get_custom_densenet_121_model(), device=device)
        type = "classification"
    elif (model_type == "Densenet169"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        model = CustomAnomalyClassifier(model=get_custom_densenet_169_model(), device=device)
        type = "classification"
    elif (model_type == "Densenet201"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        model = CustomAnomalyClassifier(model=get_custom_densenet_201_model(), device=device)
        type = "classification"
    elif (model_type == "Ensemble"):
        dataset = LesionClassificationDataset(data_path, image_dir)
        model = [CustomAnomalyClassifier(model=get_custom_densenet_121_model(), device=device),
                CustomAnomalyClassifier(model=get_custom_densenet_169_model(), device=device),
                CustomAnomalyClassifier(model=get_custom_densenet_201_model(), device=device)]
        type = "ensemble"
    if (model_path != ""):
        model.load_model(model_path)

    if (type == "detection"):
        for idx in range(30, 201, 5):
            image, target = dataset[idx]
            boxes, labels = target["boxes"], [label.cpu().detach().numpy() for label in target["labels"]]
            
            image_id, og_image = dataset.get_image(idx)
            print("Og image id:", image_id)
            show_object_detection_image(og_image, boxes, labels, f'test_image/{idx}_original_image.png')

            image = [image.to(device)]
            prediction = model.predict(image)

            pred_boxes, pred_labels = prediction[0]["boxes"].cpu().detach().numpy(), prediction[0]["labels"].cpu().detach().numpy()
            show_object_detection_image(og_image, pred_boxes, pred_labels, f'test_image/{idx}_prediction_image.png')
    elif (type == "classification"):
        for idx in range(100):
            image, target = dataset[idx]
            classes = target["classes"]

            image_id, og_image = dataset.get_image(idx)
            print("Og image id:", image_id)
            show_object_detection_image(og_image, [], [], f'classification_testing/{idx}_image.png')
            
            image = torch.stack([image.to(device)]).to(device)
            prediction = model.predict(image)

            print(f"Label: {classes}")
            print(f"Prediction: {prediction}")

        