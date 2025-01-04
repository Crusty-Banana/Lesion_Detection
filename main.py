from train import train_model_with_dataset, evaluate_model_with_dataset

train_model_with_dataset(data_path="data/annotations/train.csv",
                            image_dir="data/train_images",
                            model_path="",
                            batch_size=16, num_workers=16, device="cuda:1")