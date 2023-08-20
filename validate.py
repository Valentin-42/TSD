from ultralytics import YOLO
import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Read a txt file and plot the bounding boxes on the image")
    parser.add_argument("--fld", default="./sliced/cache/",required=False, help="Path to the txt file")
    parser.add_argument("--save_fld", default="./sliced/cache/", required=False, help="Target Folder")
    
    args = parser.parse_args()
    return args

def get_all_models(fld_path) :
    res = {}
    for exp in os.listdir(fld_path) :
        for ckp in os.listdir(os.path.join(fld_path,exp,'weights')) :
            if ckp == 'best.pt' :
                res[exp] = {}
                res[exp]['weights'] = os.path.join(fld_path,exp,'weights',ckp).replace('\\','/')
    print(res)
    return res

def val(models) :
    for exp in models.keys() :  
        # Load a model
        model = YOLO(models[exp]['weights'])

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        print(metrics)


if __name__ == "__main__":
    args = parse_args()
    models = get_all_models(args.fld)
    # val(models)
