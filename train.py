from ultralytics import YOLO
import yaml

def fine_tunining(path_to_last_weight,additionnal_epochs) :
    model = YOLO(path_to_last_weight)
    model.resume = True 

    # train the model
    results = model.train(
        epochs=additionnal_epochs, # number of additional epochs you want to train on
        imgsz=640,
        lr0=0.01,  # initial learning rate (i.e. SGD:1E-2, Adam:1E-3)
        lrf=0.01
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
    path_to_last_weight = ""
    additionnal_epochs = 10
    fine_tunining(path_to_last_weight,additionnal_epochs)