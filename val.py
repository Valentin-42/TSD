from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

import matplotlib.pyplot as plt


def compute_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
    
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = float(interArea / float(boxAArea + boxBArea - interArea))
        return iou

def predict_on_set_resolution(im_folder_path, an_folder_path, save_path, model_weights_path) :

    # Load a model
    model = YOLO(model_weights_path)
    print("Model loaded")

    confidence_scores = [] 
    ious = []
    resolutions = []

    for j,img_name in enumerate(os.listdir(im_folder_path)) :
        img = im_folder_path + img_name
        # Inference on the image
        results = model(img)
        image = cv2.imread(img)
        boxes = results[0].boxes
        if not len(boxes) > 0 :
             continue
        for i,box in enumerate(boxes) :
            bb = box.xywh[0]
            x,y,w,h = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
            cv2.putText(image, str(int(conf*100))+"%", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)

        with open(os.path.join(an_folder_path,img_name.split(".")[0]+".json"), "r") as json_f:
            data= json.load(json_f)

        res = data["width"] * data["height"]
        obj = data["objects"][0]
        bbox   = obj['bbox']
        bbox_ = [bbox["xmin"],bbox["ymin"],bbox["xmax"]-bbox["xmin"],bbox["ymax"]-bbox["ymin"]]

        iou = compute_iou(bbox_,bb)
        x,y,w,h = bbox_[0],bbox_[1],bbox_[2],bbox_[3]
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
        cv2.putText(image, "GT : "+str(int(iou*100))+" iou", (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        confidence_scores.append(conf)
        resolutions.append(res)
        ious.append(abs(iou))
        cv2.imwrite(os.path.join(save_path,img_name),image)
    

    sorted_indices = np.argsort(resolutions)
    sorted_res = np.asarray(resolutions)[sorted_indices]
    sorted_conf = np.asarray(confidence_scores)[sorted_indices]
    
    plt.plot(sorted_res,sorted_conf)
    plt.show()

    avg_conf = sum(confidence_scores)/len(confidence_scores)
    avg_ious = sum(ious)/len(ious)

    with open(save_path+"stats.txt","w+") as f:
        line = f"AVG confidence scores : {avg_conf}"
        f.write(line)
        line = f"AVG ious scores : {avg_ious}"
        f.write(line)

    f.close()

# Load the custom dataset
model_weights ="./datasets/runs/dry_run_100epochs/weights/best.pt"

test_set_img = "./datasets/test_sets/resolution/images/"
test_set_an = "./datasets/test_sets/resolution/labels/"

save_path = "./datasets/runs/dry_run_100epochs/inference/resolution/"
predict_on_set_resolution(test_set_img,test_set_an,save_path,model_weights)