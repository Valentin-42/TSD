from ultralytics import YOLO
import cv2
import numpy as np
import os

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
        img_o = img.copy()

        results = model(img, stream=True)
        boxes = results[0].boxes
        for i,box in enumerate(boxes) :
            bb = box.xywh[0]
            x,y,w,h = bb[0],bb[1],bb[2],bb[3]
            conf = box.conf[0].item()
            confidence_scores.append(conf)
            cv2.rectangle(img_o, (int(x), int(y)), (int(x+w), int(y+h)), (100, 200, 0), 3)
            cv2.putText(img_o, str(int(box.conf[0].item()*100)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite(output_path+str(num)+".jpg", img_o)
        num +=1

        # Check if the video is finished
        if not ret:
            break
    # Release the video capture and output video
    video.release()

if __name__ == '__main__':

    output_path = "./datasets/runs/dry_run_100epochs/inference/"
    
    ranging = "C:/Users/valen/Desktop/testing/New Unity Project/Recordings/1080p_60fps_30mps/"
    ranging_480 = "C:/Users/valen/Desktop/testing/New Unity Project/Recordings/480p_60fps_60mps/" 
    model_weights = "./datasets/runs/dry_run_100epochs/weights/best.pt"
    predict_on_images(ranging_480,"./datasets/runs/dry_run_100epochs/inference/","./datasets/runs/dry_run_100epochs/weights/best.pt")
    
    video_path = "./datasets/test_sets/Ranging/IMG_2741.MOV"
    # predict_on_video(video_path,60,model_weights,output_path)