# evaluate_model_with_dataset(data_path="data/annotations/test.csv", 
#                             image_dir="data/test_images",
#                             model_path="models/beta_model",
#                             batch_size=10, num_workers=4, device="cuda:0")

import argparse
from train import (train_model_with_dataset, 
                   evaluate_model_with_dataset,
                   inference_model)

def main(args):
    if args.action == "train":
        if args.model == "Densnet121":
            train_model_with_dataset(data_path="data/annotations/train.csv",
                                    image_dir="data/train_images",
                                    model_type="Densnet121",
                                    model_path=args.model_path,
                                    batch_size=6, num_workers=16, device=args.device,
                                    device_ids=[0, 1, 3],
                                    epochs=args.epoch, learning_rate=0.000000001,
                                    checkpoint_path="models/densenet_121")  
        elif args.model == "Densenet169":
            train_model_with_dataset(data_path="data/annotations/train.csv",
                                    image_dir="data/train_images",
                                    model_type="Densenet169",
                                    model_path=args.model_path,
                                    batch_size=6, num_workers=16, device=args.device,
                                    device_ids=[0, 1, 3],
                                    epochs=args.epoch, learning_rate=0.000000001,
                                    checkpoint_path="models/densenet_169")
        elif args.model == "Densenet201":
            train_model_with_dataset(data_path="data/annotations/train.csv",
                                    image_dir="data/train_images",
                                    model_type="Densnet201",
                                    model_path=args.model_path,
                                    batch_size=6, num_workers=16, device=args.device,
                                    device_ids=[0, 1, 3],
                                    epochs=args.epoch, learning_rate=0.000000001,
                                    checkpoint_path="models/densenet_201")  
        elif args.model == "FasterRCNN":
            train_model_with_dataset(data_path="data/annotations/train.csv",
                                    image_dir="data/train_images",
                                    model_type="FasterRCNN",
                                    model_path=args.model_path,
                                    batch_size=6, num_workers=16, device=args.device,
                                    device_ids=[0, 1, 3],
                                    epochs=args.epoch, learning_rate=0.000000001,
                                    checkpoint_path="models/faster_rcnn") 
        elif args.model == "RetinaNet":
            train_model_with_dataset(data_path="data/annotations/train.csv",
                                    image_dir="data/train_images",
                                    model_type="RetinaNet",
                                    model_path=args.model_path,
                                    batch_size=6, num_workers=16, device=args.device,
                                    device_ids=[0, 1, 3],
                                    epochs=args.epoch, learning_rate=0.000000001, 
                                    checkpoint_path="models/retinanet")  
    elif args.action == "inference":
        inference_model(data_path="data/annotations/train.csv",
                                image_dir="data/train_images",
                                model_type=args.model,
                                model_path=args.model_path,
                                device=args.device)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesion Detection")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference', 'test'], help='Action to perform: train or inference or validation')
    
    parser.add_argument('--model', type=str, default='FasterRCNN', choices=['FasterRCNN', 'RetinaNet', 'Densenet121', 'Densenet169', 'Densenet201'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epoch', type=str, default='1')

    args = parser.parse_args()
    main(args)

    # For Inference