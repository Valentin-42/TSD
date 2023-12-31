from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

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

def predict_on_images(folder_path, save_path, model_weights_path) :

    # Load a model
    model = YOLO(model_weights_path)
    print("Model loaded")

    confidence_scores = [] 

    for j,img_name in enumerate(os.listdir(folder_path)) :
        img = folder_path + img_name
        # Inference on the image
        results = model(img)
        image = cv2.imread(img)
        boxes = results[0].boxes
        for i,box in enumerate(boxes) :
            bb = box.xywh[0]
            x,y,w,h = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            confidence_scores.append(conf)
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
            cv2.putText(image, str(int(conf*100)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.imwrite(save_path+img_name, image)
    
    avg_conf = sum(confidence_scores)/len(confidence_scores)

    with open(save_path+"stats.txt","w+") as f:
        line = f"AVG confidence scores : {avg_conf}"
        f.write(line)

def predict_on_video(video_path,FPS,model_weights_path,output_path) :
    print(output_path)
    # Read the video
    video = cv2.VideoCapture(video_path)
    print("Video loaded")

    # Load a model
    model = YOLO(model_weights_path)
    print("Model loaded")

    confidence_scores = [] 

    num = 0
    while True:
        print(num)
        # Get a frame from the video
        ret, img = video.read()
        # Check if the video is finished
        if not ret:
            break

        img_o = img.copy()

        results = model(img)
        boxes = results[0].boxes
        for i,box in enumerate(boxes) :
            bb = box.xywh[0]
            x,y,w,h = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            confidence_scores.append(conf)
            cv2.rectangle(img_o, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
            cv2.putText(img_o, str(int(box.conf[0].item()*100)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(output_path, str(num)+".jpg"), img_o)
        num +=1

    avg_conf = sum(confidence_scores)/len(confidence_scores)
    with open(os.path.join(output_path,"stats.txt"),"w+") as f:
        line = f"AVG confidence scores : {avg_conf}"
        f.write(line)

    # Release the video capture and output video
    video.release()

def predict_vs_resolution() :
    with open("multi_res.json", "r") as f:
        data = json.load(f)
    
    for key, value in data.items():
        print(key, value)
        


if __name__ == '__main__':

    
    ranging = "C:/Users/valen/Desktop/testing/New Unity Project/Recordings/1080p_60fps_30mps/"
    ranging_480 = "C:/Users/valen/Desktop/testing/New Unity Project/Recordings/480p_60fps_60mps/" 

    model = "train2" # dry_run_100epochs
    output_path = f"./datasets/runs/{model}/inference/"
    model_weights = "./datasets/train2/_tune_1211e_00000_0_copy_paste=0.8000,mosaic=0.3000,scale=0.3000_2023-07-23_11-12-39/yolov8n.pt"

    # model_weights = f"./datasets/runs/{model}/weights/best.pt"
    # predict_on_images(ranging_480,"./datasets/runs/dry_run_100epochs/inference/","./datasets/runs/dry_run_100epochs/weights/best.pt")
    
    video_path = "./datasets/test_sets/Ranging/IMG_2741.MOV"

    video640   = "./datasets/test_sets/Ranging/640x640/movie_008.mp4"
    video1024  = "./datasets/test_sets/Ranging/1024x1024/movie_007.mp4"
    video4k  = "./datasets/test_sets/Ranging/2560x1440/movie_010.mp4"

    videos = [video640,video1024,video4k]
    output_fld_names = ["640","1024","1440"]
    for i,video in enumerate(videos) :
        print(video)
        if not os.path.exists(os.path.join(output_path,output_fld_names[i])):
            os.mkdir(os.path.join(output_path,output_fld_names[i]))
        predict_on_video(video,60,model_weights,os.path.join(output_path,output_fld_names[i]))