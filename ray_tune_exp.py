from ultralytics import YOLO
import yaml
import ray
from ray import tune
from ray.tune.trainable.function_trainable import wrap_function
from ray.tune.tuner import Tuner, TuneConfig




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

    param_space={
            "copy_paste": tune.choice([0.0,0.4,0.8]),
            "scale": tune.choice([0.0,0.4, 0.8]),
            "mosaic": tune.choice([0.0,0.4, 0.8]),
            "imgsz": tune.choice([416, 608, 800]),
        }
    
    model = YOLO("yolov8n.pt")
    data_path="./configs/nano/data.yaml"
    ray.shutdown()
    ray.init()

    # Run Ray Tune on the model
    result_grid = model.tune(data=data_path,
                            space=param_space,
                            gpu_per_trial=1,
                            epochs=50)
    
    print("   >>>>>    ")
    print(result_grid)
    print("   >>>>>    ")
    txt_file_path = "results.txt"
    with open(txt_file_path, 'w') as text_file:
        text_file.write(result_grid)

    ray.shutdown()