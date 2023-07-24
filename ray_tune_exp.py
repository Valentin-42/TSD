from ultralytics import YOLO
import yaml
import ray
from ray import tune
from ray.tune.trainable.function_trainable import wrap_function
from ray.tune.tuner import Tuner, TuneConfig


if __name__ == "__main__":

    param_space={
            "copy_paste": tune.choice([0.8]),
            "scale": tune.choice([0.8]),
            "mosaic": tune.choice([0.8]),
            "imgz": tune.choice([640, 1280]),
        }
    
    model = YOLO("yolov8n.pt")
    data_path="/mnt/gpu_storage/traffic-sign-detection/TSD/configs/nano/data.yaml"

    ray.shutdown()
    ray.init()

    # Run Ray Tune on the model
    result_grid = model.tune(data=data_path,
                            space=param_space,
                            gpu_per_trial=2,
                            max_samples= 4,
                            epochs=50,
                            batch=32)
    
    print("   >>>>>    ")
    print(result_grid)
    print("   >>>>>    ")
    txt_file_path = "results.txt"
    with open(txt_file_path, 'w') as text_file:
        text_file.write(result_grid)

    ray.shutdown()