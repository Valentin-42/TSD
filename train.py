from ultralytics import YOLO
import yaml
from ray import tune

def RayTune() :
    data_path = "./configs/nano/data.yaml"
    model = YOLO("yolov8n.pt")
    # Run Ray Tune on the model
    result_grid = model.tune(data=data_path,
                            space={"lr0":tune.uniform(1e-5, 1e-1)},
                            epochs=10,
                            results_dir="results/LR0/"
                            )
    

def fine_tunining(path_to_last_weight,additionnal_epochs) :
    model = YOLO(path_to_last_weight)
    model.resume = True 

    # train the model
    results = model.train(
        # resume=True,
        epochs=additionnal_epochs, # number of additional epochs you want to train on
        imgsz=640,
        lr0=0.01,  # initial learning rate (i.e. SGD:1E-2, Adam:1E-3)
        lrf=0.01,
        save_period = 10,
    )

def start_training_scratch() :
    params_path = "./configs/nano/params.yaml"

    #Loading model config
    print("Loading Config ...")
    with open(params_path, 'r') as f:
        args = yaml.safe_load(f)
    print("Config Loaded !")

    
    # Loading model
    model = args["model"]
    print(f"Using {model}")

    model = YOLO(model)  # load a pretrained model

    # Train the model
    print(">> Starting training <<")
    print(str(args))
    results = model.train(**args)

if __name__ == "__main__":
    # start_training_scratch()

    path_to_last_weight = "./runs/detect/train/weights/last.pt"
    additionnal_epochs = 100
    fine_tunining(path_to_last_weight,additionnal_epochs)