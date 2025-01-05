from train import train_model_with_dataset, evaluate_model_with_dataset

train_model_with_dataset(data_path="data/annotations/train.csv",
                            image_dir="data/train_images",
                            model_path="",
                            batch_size=10, num_workers=4, device="cuda:0",
                            device_ids=[0, 1, 3], #currently doesnt work
                            epochs=10, learning_rate=0.000000001,
                            checkpoint_path="models/beta_model")  