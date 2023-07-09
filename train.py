from ultralytics import YOLO
import yaml

if __name__ == "__main__":
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
    print(str(**args))
    results = model.train(**args)