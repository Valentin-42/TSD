from ultralytics import YOLO
import yaml
from ray import tune

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
    model.epochs   = 50
    model.imgsz    = 50
    model.save_period = 10
    model.data     = path_to_config
    model.device   = 0
    model.exist_ok = True
    model.batch    = 128
    model.project_name = "Optimizer_Tuning" 
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
        model.name = exp
        model.optimizer = experiments[exp]["optimizer"]
        model.lr0 = experiments[exp]["lr0"]
        model.lrf = experiments[exp]["lrf"]
        results = model.train()



def hpp_tuning(path_to_weights, path_to_config) :

    model = YOLO(path_to_weights)

    # Default params
    model.epochs   = 50
    model.imgsz    = 50
    model.save_period = 10
    model.data     = path_to_config
    model.device   = 0
    model.exist_ok = True
    model.batch    = 128
    model.project_name = "Hyperparams_Tuning" 
    model.optimizer = 'SGD'

    # 
    curve = [0.0, 0.35, 0.65, 1.0]
    i,j,k,l = 0
    for exp in range(4**4) : 
        model.name = f"exp_{exp}"
        model.mosaic = curve[i] 
        model.mixup = curve[j]
        model.copy_paste = curve[k]
        model.scale = curve[l]

        results = model.train()
        i+= 1
        j+= round(i/4)
        k+= round(j/4)
        l+= round(l/4)

        if i == 4 :
            i=0
        elif j ==4:
            j=0
        elif k==4:
            k=0
        elif l==4 :
            l=0





if __name__ == "__main__":
    start_training_scratch()

    # path_to_last_weight = "./runs/detect/train/weights/last.pt"
    # additionnal_epochs = 100
    # fine_tunining(path_to_last_weight,additionnal_epochs)

    # RayTune()