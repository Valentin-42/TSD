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
    
    # Create bins/categories for resolutions
    resolution_bins = [0, 50000, 100000, 200000, 300000, np.inf]  # Customize the bin ranges as needed

    # Initialize lists to store the total confidence scores and counts for each bin
    total_confidence_per_bin = [0] * len(resolution_bins)
    counts_per_bin = [0] * len(resolution_bins)

    # Group resolutions into the bins
    for resolution, confidence_score in zip(resolutions, confidence_scores):
        for i in range(len(resolution_bins) - 1):
            if resolution_bins[i] <= resolution < resolution_bins[i + 1]:
                total_confidence_per_bin[i] += confidence_score
                counts_per_bin[i] += 1
                break

    # Calculate average confidence score for each bin
    avg_confidence_per_bin = [total_confidence / count if count > 0 else 0 for total_confidence, count in zip(total_confidence_per_bin, counts_per_bin)]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(resolution_bins) - 1), avg_confidence_per_bin)
    plt.xlabel('Resolution Categories')
    plt.ylabel('Average Confidence Score')
    plt.title('Average Confidence Score per Resolution Category')
    plt.xticks(range(len(resolution_bins) - 1), ['0-50K', '50K-100K', '100K-200K', '200K-300K', '300K+'])  # Customize labels if needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)
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