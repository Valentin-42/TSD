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

def predict_on_set(im_folder_path, an_folder_path, save_path, model_weights_path, plot=False):
    # Load a model
    model = YOLO(model_weights_path)
    print("Model loaded")

    confidence_scores = [] 
    ious = []
    no_detections = []
    TP = [] # GT and DT
    FN = [] # No dt but GT
    FP = [] # Dt but not GT

    resolutions = {}

    for j,img_name in enumerate(os.listdir(im_folder_path)) :
        img = im_folder_path + img_name
        # Inference on the image
        results = model(img)
        image = cv2.imread(img)
        boxes = results[0].boxes

        with open(os.path.join(an_folder_path,img_name.split(".")[0]+".json"), "r") as json_f:
            data= json.load(json_f)

        res = data["width"] * data["height"]
        resolutions[res] = [0,0,0] # TP FP FN
        
        obj = data["objects"][0]
        bbox   = obj['bbox']
        bbox_ = [bbox["xmin"],bbox["ymin"],bbox["xmax"]-bbox["xmin"],bbox["ymax"]-bbox["ymin"]]
        x,y,w,h = bbox_[0],bbox_[1],bbox_[2],bbox_[3]

        if len(boxes) == 0  : # No DT 
            print("No DT but GT")
            FN.append(1) # No DT but GT
            cv2.putText(image, "Not Detected", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            resolutions[res][1] = resolutions[res][1] + 1

        elif len(boxes) > 1 : #Wrong DT
            iou = 0
            idx = None
            for i,box in enumerate(boxes) :
                bb = box.xywh[0]
                iou_ = compute_iou(bbox_,bb)
                if iou_ > iou : # Find max iou
                    iou = iou_
                    idx = i

            if idx == None : # If none of the dt are overlapping
                FP.append(1) # DT but no GT
                resolutions[res][2] = resolutions[res][2] + 1 
            else : # One overlap so one DT with GT 
                for i,box in enumerate(boxes) :
                    iou_ = compute_iou(bbox_,bb)
                    if iou_ == iou :
                        ious.append(iou_)  # we found the corresponding DT
                        conf = box.conf[0].item()
                        confidence_scores.append(conf)
                        bb = box.xywh[0]
                        xd,yd,wd,hd = bb[0],bb[1],bb[2],bb[3]
                        cv2.rectangle(image, (int(xd), int(yd)), (int(xd+wd), int(yd+hd)), (100, 200, 0), 3)
                        cv2.putText(image, str(int(conf*100))+"%", (int(xd), int(yd)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
                        TP.append(1)
                        resolutions[res][0] = resolutions[res][0] + 1 
                    else :
                        FP.append(1) #DT but No GT
                        resolutions[res][2] = resolutions[res][2] + 1 
                        confidence_scores.append(box.conf[0].item())

        else : # Good 1 DT 
            box = boxes[0]
            bb = box.xywh[0]
            iou = compute_iou(bbox_,bb)
            xd,yd,wd,hd = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            confidence_scores.append(conf)
            cv2.rectangle(image, (int(xd), int(yd)), (int(xd+wd), int(yd+hd)), (100, 200, 0), 3)

            if (iou == 0) : #Check if not well placed
                FP.append(1)
                resolutions[res][2] = resolutions[res][2] + 1
                cv2.putText(image, str(int(conf*100))+"%", (int(xd), int(yd)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
                cv2.putText(image, "Missed", (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else : # Nice placed
                TP.append(1)
                resolutions[res][0] = resolutions[res][0] + 1
                confidence_scores.append(conf)
                ious.append(abs(iou))
                cv2.putText(image, str(int(conf*100))+"%", (int(xd), int(yd)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
                cv2.putText(image, "GT : "+str(int(iou*100))+" iou", (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path,img_name),image)
    
    print(resolutions)
    avg_conf = sum(confidence_scores)/(len(confidence_scores)+1e-6)
    avg_ious = sum(ious)/(len(ious)+1e-6)

    accuracy  = sum(TP) / (sum(TP) + sum(FP) + sum(FN) +1e-6) *100
    precision = sum(TP) / (sum(TP) + sum(FP) +1e-6) *100
    recall    = sum(TP) / (sum(TP) + sum(FN) +1e-6) *100

    with open(save_path+"stats.txt","w+") as f:
        line = f"AVG confidence scores : {avg_conf} \n"
        f.write(line)
        line = f"AVG ious scores : {avg_ious} \n"
        f.write(line)
        line = f"accuracy : {accuracy} %\n"
        f.write(line)
        line = f"precision : {precision} %\n"
        f.write(line)
        line = f"recall : {recall} %\n"
        f.write(line)
    f.close()


    def hist() :
        keys = list(resolutions.keys())
        keys.sort()

        tp = []
        fn = []
        fp = []
        for key in keys :
            tp.append(resolutions[key][0])
            fn.append(resolutions[key][1])
            fp.append(resolutions[key][2])

        plt.scatter(keys,tp)
        plt.bar(keys, tp, align='center', alpha=0.5, color='blue', width=0.1, edgecolor='blue')
        plt.show()


    hist()




# Load the custom dataset

for model in ["train","train2"] : #dry_run_100epochs

    model_weights = f"./datasets/runs/{model}/weights/best.pt"
    if model == "train2" :
        model_weights = "./datasets/train2/_tune_1211e_00000_0_copy_paste=0.8000,mosaic=0.3000,scale=0.3000_2023-07-23_11-12-39/yolov8n.pt"
    #ambiguous occluded
    for set in ["resolution","ambiguous", "occluded"] : 

        test_set_img = f"./datasets/test_sets/{set}/images/"
        test_set_an =  f"./datasets/test_sets/{set}/labels/"

        save_path = f"./datasets/runs/{model}/inference/{set}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # predict_on_set_resolution(test_set_img,test_set_an,save_path,model_weights)
        if set == "resolution" :
            plot=True
        else :
            plot=False
            print("end res")
        predict_on_set(test_set_img, test_set_an, save_path, model_weights,plot)
        break