from ultralytics import YOLO
import yaml
from ray import tune
import argparse

def RayTune() :
    data_path = "/mnt/gpu_storage/traffic-sign-detection/TSD/configs/nano/data.yaml"
    model = YOLO("yolov8n.pt")
    # Run Ray Tune on the model
    result_grid = model.tune(data=data_path,
                            space={"lr0":tune.uniform(1e-5, 1e-1)},
                            epochs=10,
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


def optimizer_tuning(path_to_weights, path_to_config) :
    model = YOLO(path_to_weights)

    # Default params
    epochs   = 200
    imgsz    = 640
    save_period = 10
    data     = path_to_config
    device   = 0
    exist_ok = True
    batch    = 128
    project_name = "Optimizer_Tuning" 
    # 
    experiments = {
            'Opt-SGD': {'optimizer':'SGD', 'lr0':0.01, 'lrf':0.01},
            'Opt-Adam': {'optimizer':'Adam', 'lr0':0.01, 'lrf':0.01},
            'Opt-SGD2': {'optimizer':'SGD', 'lr0':0.02, 'lrf':0.01},
            'Opt-Adam2': {'optimizer':'Adam', 'lr0':0.02, 'lrf':0.01},
            'Opt-SGD5': {'optimizer':'SGD', 'lr0':0.05, 'lrf':0.01},
            'Opt-Adam5': {'optimizer':'Adam', 'lr0':0.05, 'lrf':0.01},
        }

    for exp in experiments.keys() : 
        name = exp
        optimizer = experiments[exp]["optimizer"]
        lr0 = experiments[exp]["lr0"]
        lrf = experiments[exp]["lrf"]

        results = model.train(
            data = data,
            epochs = epochs,
            imgsz = imgsz,
            save_period = save_period,
            device = device,
            exist_ok = exist_ok,
            batch = batch,
            project = project_name,
            name = name,
            patience = epochs,
            optimizer = optimizer,
            lr0 = lr0,
            lrf = lrf
        )

def hpp_tuning(path_to_weights, path_to_config, epochs) :


    # Default params
    epochs   = epochs
    imgsz    = 640
    patience = epochs
    save_period = 10
    data     = path_to_config
    device   = 0
    exist_ok = True
    batch    = 128
    project_name = "Hyperparams_Tuning" 
    optimizer = 'SGD'

    curve = [0.0, 0.35, 0.65, 1.0]
    i,j,k,l = 0,0,0,0
    for exp in range(256) : 
        model = YOLO(path_to_weights)

        name = f"exp_{exp}_{i}_{j}_{k}_{l}"
        mosaic = curve[i] 
        mixup = curve[j]
        copy_paste = curve[k]
        scale = curve[l]

        results = model.train(
            data = data,
            epochs = epochs,
            patience = patience,
            imgsz = imgsz,
            save_period = save_period,
            device = device,
            exist_ok = exist_ok,
            batch = batch,
            project = project_name,
            name = name,
            optimizer = optimizer,
            mosaic = mosaic,
            mixup = mixup,
            copy_paste = copy_paste,
            scale = scale,
            resume = False
        )
        i+= 1
        if i == 3 :
            i=0
            j+=1
        elif j ==3:
            j=0
            k+=1
        elif k==3:
            k=0
            l+=1
        elif l==3 :
            l=0
            break


def final_training(path_to_weights, path_to_config) :

    model = YOLO(path_to_weights)

    # Default params
    epochs   = 500
    imgsz    = 640
    save_period = 100
    data     = path_to_config
    device   = 0
    exist_ok = True
    batch    = 128
    project_name = "Final_training"
    name = 'exp'
    optimizer = 'SGD', 
    lr0 =float(0.01),
    lrf =float(0.01),
    mosaic = 0,
    mixup  = 0.4,
    copy_paste = 0.7,
    scale = 0.7,
    resume = False
    # 

    results = model.train(
        data = data,
        epochs = epochs,
        imgsz = imgsz,
        save_period = save_period,
        device = device,
        exist_ok = exist_ok,
        batch = batch,
        project = project_name,
        name = name,
        patience = epochs,
        optimizer = optimizer,
        lr0 = lr0,
        lrf = lrf,
        mosaic = mosaic,
        mixup = mixup,
        copy_paste = copy_paste,
        scale = scale,
        resume = False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="./configs/nano/yolov8n.pt", help = "path to .pt")
    parser.add_argument("-c", default="./configs/nano/data_4classes.yaml", help = "data file path")
    parser.add_argument("-e", default=100, help = "nbs of epochs")

    args = parser.parse_args()
    # optimizer_tuning(args.w, args.c)
    # hpp_tuning(args.w, args.c, args.e)
    final_training(args.w, args.c)
