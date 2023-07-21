from ultralytics import YOLO
import yaml
import ray
from ray import tune
from ray.tune.trainable.function_trainable import wrap_function


@wrap_function
def train_yolov8(config):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Set the hyperparameters from the config dictionary
    model.copy_paste = config["copy_paste"]
    model.scale = config["scale"]
    model.mosaic = config["mosaic"]
    model.imgsz = config["imgsz"]

    # Train the model
    data="./configs/nano/data.yaml"
    model.train(data,epochs=100)

if __name__ == "__main__":

    # Create a Ray Tune experiment
    tune_experiment = ray.tune.Experiment(
        name="yolov8-copy_paste-scale-mosaic-imgsz-tuning",
        function=train_yolov8,
        metric="val_loss",
        mode="min",
        resources_per_trial={"gpu": 1},
        num_samples=10,
        config={
            "copy_paste": tune.choice([0.0,0.4,0.8]),
            "scale": tune.choice([0.0,0.4, 0.8]),
            "mosaic": tune.choice([0.0,0.4, 0.8]),
            "imgsz": tune.choice([416, 608, 800]),
        }
    )

    # Tune the hyperparameters
    tune_experiment.run()