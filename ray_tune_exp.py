from ultralytics import YOLO
import yaml
import ray
from ray import tune
from ray.tune.function_runner import wrap_function

@wrap_function
def train_yolov8(data, copy_paste, scale, mosaic):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Set the copy_paste augmentation parameter
    model.copy_paste = copy_paste

    # Set the scale augmentation parameter
    model.scale = scale

    # Set the mosaic augmentation parameter
    model.mosaic = mosaic

    # Train the model
    data="./configs/nano/data.yaml"
    model.train(data,epochs=100)

if __name__ == "__main__":

    # Create a Ray Tune experiment
    tune_experiment = ray.tune.Experiment(
        name="yolov8-copy_paste-scale-mosaic-tuning",
        function=train_yolov8,
        metric="val_loss",
        mode="min",
        resources_per_trial={"gpu": 1},
        num_samples=10,
    )

    # Tune the copy_paste, scale, and mosaic augmentation parameters
    tune_experiment.run(
        {
            "copy_paste": tune.choice([0.0,0.4,0.8]),
            "scale": tune.choice([0.0,0.4, 0.8]),
            "mosaic": tune.choice([0.0,0.4, 0.8]),
        }
    )
