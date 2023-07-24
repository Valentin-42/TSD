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
    no_detections = []

    for j,img_name in enumerate(os.listdir(im_folder_path)) :
        img = im_folder_path + img_name
        # Inference on the image
        results = model(img)
        image = cv2.imread(img)
        boxes = results[0].boxes

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
        x,y,w,h = bbox_[0],bbox_[1],bbox_[2],bbox_[3]


        if len(boxes) == 0 : # No DT
            print("NO DT")
            confidence_scores.append(0)
            resolutions.append(res)
            ious.append(0)
            no_detections.append(1)
            iou = 0
        else :
            iou = compute_iou(bbox_,bb)
            confidence_scores.append(conf)
            resolutions.append(res)
            no_detections.append(0)
            ious.append(abs(iou))
            cv2.putText(image, "GT : "+str(int(iou*100))+" iou", (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(save_path,img_name),image)
    

    sorted_indices = np.argsort(resolutions)
    sorted_res = np.asarray(resolutions)[sorted_indices]
    sorted_conf = np.asarray(confidence_scores)[sorted_indices]
    sorted_ious = np.asarray(ious)[sorted_indices]

    sorted_no_dt = np.asarray(no_detections)[sorted_indices]

    plt.scatter(sorted_res,sorted_conf)
    plt.bar(sorted_res, sorted_conf, align='center', alpha=0.5, color='gray', width=0.2, edgecolor='black')
    plt.show()

    plt.scatter(sorted_res,sorted_ious)
    plt.bar(sorted_res, sorted_ious, align='center', alpha=0.5, color='gray', width=0.2, edgecolor='black')
    plt.show()

    plt.scatter(sorted_res,sorted_no_dt)
    plt.bar(sorted_res, sorted_no_dt, align='center', alpha=0.5, color='gray', width=0.2, edgecolor='black')
    plt.show()

    avg_conf = sum(confidence_scores)/len(confidence_scores)
    avg_ious = sum(ious)/len(ious)
    success_rate = 100 - (sum(no_detections)/len(os.listdir(im_folder_path)) * 100)
    with open(save_path+"stats.txt","w+") as f:
        line = f"AVG confidence scores : {avg_conf} \n"
        f.write(line)
        line = f"AVG ious scores : {avg_ious} \n"
        f.write(line)
        line = f"Success rate : {success_rate} %\n"
        f.write(line)
    f.close()

def predict_on_set(im_folder_path, an_folder_path, save_path, model_weights_path):
        # Load a model
    model = YOLO(model_weights_path)
    print("Model loaded")

    confidence_scores = [] 
    ious = []
    no_detections = []

    for j,img_name in enumerate(os.listdir(im_folder_path)) :
        img = im_folder_path + img_name
        # Inference on the image
        results = model(img)
        image = cv2.imread(img)
        boxes = results[0].boxes

        with open(os.path.join(an_folder_path,img_name.split(".")[0]+".json"), "r") as json_f:
            data= json.load(json_f)

        res = data["width"] * data["height"]
        obj = data["objects"][0]
        bbox   = obj['bbox']
        bbox_ = [bbox["xmin"],bbox["ymin"],bbox["xmax"]-bbox["xmin"],bbox["ymax"]-bbox["ymin"]]
        x,y,w,h = bbox_[0],bbox_[1],bbox_[2],bbox_[3]


        if len(boxes) == 0  : # No DT 
            print("NO DT")
            confidence_scores.append(0)
            ious.append(0)
            no_detections.append(1)
            iou = 0
            cv2.putText(image, "Missed", cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif len(boxes) > 1 : #Wrong DT
            iou = 0
            idx = None
            for i,box in enumerate(boxes) :
                bb = box.xywh[0]
                iou_ = compute_iou(bbox_,bb)
                if iou_ > iou :
                    iou = iou_
                    idx = i
            if idx == None :
                no_detections.append(1)     
                ious.append(0)     
                confidence_scores.append(0)
            else :
                for i,box in enumerate(boxes) :
                    iou_ = compute_iou(bbox_,bb)
                    if iou_ == iou :
                        no_detections.append(0)   
                        ious.append(iou_)  
                        conf = box.conf[0].item()
                        confidence_scores.append(conf)
                        bb = box.xywh[0]
                        x,y,w,h = bb[0],bb[1],bb[2],bb[3]
                        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
                        cv2.putText(image, str(int(conf*100))+"%", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
                    else :
                        no_detections.append(1)   
                        ious.append(0)  
                        confidence_scores.append(0)
        else : # Good DT
            box = boxes[0]
            bb = box.xywh[0]
            x,y,w,h = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
            cv2.putText(image, str(int(conf*100))+"%", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
            iou = compute_iou(bbox_,bb)
            confidence_scores.append(conf)
            no_detections.append(0)
            ious.append(abs(iou))
            cv2.putText(image, "GT : "+str(int(iou*100))+" iou", (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path,img_name),image)
    

    avg_conf = sum(confidence_scores)/len(confidence_scores)
    avg_ious = sum(ious)/len(ious)
    success_rate = 100 - (sum(no_detections)/len(os.listdir(im_folder_path)) * 100)
    with open(save_path+"stats.txt","w+") as f:
        line = f"AVG confidence scores : {avg_conf} \n"
        f.write(line)
        line = f"AVG ious scores : {avg_ious} \n"
        f.write(line)
        line = f"Success rate : {success_rate} %\n"
        f.write(line)
    f.close()



# Load the custom dataset
model_weights ="./datasets/runs/dry_run_100epochs/weights/best.pt"

#ambiguous occluded
set = "ambiguous" #ambiguous occluded resolution
set = "occluded"

test_set_img = f"./datasets/test_sets/{set}/images/"
test_set_an =  f"./datasets/test_sets/{set}/labels/"

save_path = f"./datasets/runs/dry_run_100epochs/inference/{set}/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
# predict_on_set_resolution(test_set_img,test_set_an,save_path,model_weights)
predict_on_set(test_set_img, test_set_an, save_path, model_weights)