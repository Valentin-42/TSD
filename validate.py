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
    # print(res)
    return res

def val(models) :
    df = pd.DataFrame()
    d = {
            "mAP@50-95": [],
            "mAP@50": [],
            "mAP@75": [],
            "mP": [],
            "mR": []
        }
    
    for exp in models.keys() :  
        # Load a model
        model = YOLO(models[exp]['weights'])

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered

        # Create a pandas dataframe
        d["mAP@50-95"].append(metrics.box.map)
        d["mAP@50"].append(metrics.box.map50)
        d["mAP@75"].append(metrics.box.map75)
        d["mP"].append(metrics.box.mp)
        d["mR"].append(metrics.box.mr)

        # Print the dataframe
        print(df)
        break

if __name__ == "__main__":
    args = parse_args()
    models = get_all_models(args.fld)
    val(models)
