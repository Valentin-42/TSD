import os
import torch
import re
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def ranging2(model_path, folder_path) :
    # Load pre-trained model
    model = YOLO(model_path)

    l = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
    print("Running ...")
    predictions = model(l)
    print("Done !!")
    confidence_scores = []
    for i,pred in enumerate(predictions) :
        pred = pred.boxes.conf.numpy()
        print(f"img : {i} > {len(pred)}")
        if pred == None : 
            confidence_scores.append(0)
        elif len(pred) == 0 :
            confidence_scores.append(0)
        else :
            confidence_scores.append(sum(pred)/len(pred))

    image_indices = np.arange(len(confidence_scores))
    x_smooth = np.linspace(image_indices.min(), image_indices.max(), 300)
    spl = make_interp_spline(image_indices, confidence_scores, k=100)
    y_smooth = spl(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', linewidth=3, label='Smooth')

    # Plot AP versus distance
    plt.plot(image_indices, confidence_scores, marker='o')
    plt.xlabel('Distance (m)')
    plt.ylabel('Confidence Score')
    plt.title('Confidence versus Distance (YOLO)')
    plt.grid()
    plt.show()


def ranging(model_path, folder_path) :
        
    # Load pre-trained model
    model = YOLO(model_path)

    # Collect predictions and ground truths
    predictions = []

    l = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
    cfs = []
    # Iterate through images
    for filename in l:
        print(filename)
        image_path = os.path.join(folder_path, filename)
        predictions = model(image_path)
        confidence_scores = []
        for pred in predictions :
            confidence_scores.append(pred.conf)
        if len(confidence_scores) == 0 :
            cfs.append(sum(confidence_scores)/len(confidence_scores))
        else :
            cfs.append(0)

    image_indices = np.arange(len(confidence_scores))
    # Plot AP versus distance
    plt.plot(image_indices, cfs, marker='o')
    plt.xlabel('Distance (m)')
    plt.ylabel('Average Precision')
    plt.title('AP versus Distance (YOLO)')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Set your paths and model parameters
    model_path = './configs/nano/yolov8n.pt'
    folder_path = './datasets/test_sets/ranging/640x640/'
    print("Starting ...")
    ranging2(model_path,folder_path)