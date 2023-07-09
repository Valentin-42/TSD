import json
import os
from PIL import Image, ImageDraw, ImageColor, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def dashboard(dataframe) :
    st.title('Uber pickups in NYC')

    st.dataframe(data=dataframe)

def load_annotation(image_path):
    with open(image_path, 'r') as fid:
        anno = json.load(fid)
    return anno

def parser(data, objects, images,id) :
    # Extract image resolution
    images.append({"Id":id,"Width":data["width"],"Height":data["height"]})
    obj_dict = {}
    for obj in data["objects"] :

        obj_dict = {
            "Label" : obj["label"],
            "Width" : abs(obj["bbox"]["xmax"] - obj["bbox"]["xmin"]),
            "Height": abs(obj["bbox"]["ymax"] - obj["bbox"]["ymin"]),
            "Size"  : (obj["bbox"]["xmax"] - obj["bbox"]["xmin"]) * (obj["bbox"]["ymax"] - obj["bbox"]["ymin"])
        }
        objects.append(obj_dict)
    
def show_class_distrib(class_distribution) :

    print(f"Number of classes found : {len(class_distribution)}")

    fig, ax = plt.subplots()
    class_distribution.plot(kind='bar', ax=ax)
    plt.title('Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')

    # Replace x-axis labels with numbers
    x_tick_positions = range(len(class_distribution))
    x_tick_labels = class_distribution.index
    plt.xticks(x_tick_positions, x_tick_positions, rotation=45, fontsize=5)
    plt.show()

def show_size_distrib(size_category_distribution):
    size_category_distribution.plot(kind='bar')
    plt.title('Size Distribution')
    plt.xlabel('object size')
    plt.ylabel('Count')
    plt.show()

if __name__ == '__main__':

    objects = [] #List of dict containing objects properties (Label, Size)
    images  = [] #List of dict containing images properties  (Width, Height)
    # annotations_path = "./MTSD/extracted/mtsd_v2_fully_annotated/annotations/"
    # annotations_path = "./datasets/default_MTSD/val/labels/"
    annotations_path = "./datasets/default_MTSD/test/images/"
    
    N = len(os.listdir(annotations_path))
    mode=0
    if mode != 0 :
        print(f"Folder :{annotations_path}")
        print(f"Number of image in folder :{N}")
        for id,f in enumerate(os.listdir(annotations_path)) :
            # load the annotation json
            path = os.path.join(annotations_path,f)
            anno = load_annotation(path)
            parser(anno,objects,images,id)

            print(f" --> {np.round((id/N)*100,2)} % <--", end="\r")
            # if id == 1000 :
            #     break 

        # Create a pandas DataFrame from lists
        df_objects = pd.DataFrame(objects)
        df_images  = pd.DataFrame(images)

        # Print object sizes
        print("Object Sizes:")
        print(df_objects[["Label", "Size"]])
        df_objects.plot(x='Height', y='Width', kind='scatter', title='Objects Width vs Height') #Plot 

        # Print images sizes
        print("Images Sizes:")
        print(df_images[["Id", "Width","Height"]])
        df_images.plot(x='Height', y='Width', kind='scatter', title='Images Width vs Height') #Plot 

        # Split the 'Size' column into bins, creating a new cutted dataframe
        bins = [0, 1024, 9216, float('inf')]
        labels = ['small', 'medium', 'large']
        df_objects['Size Category'] = pd.cut(df_objects['Size'], bins=bins, labels=labels)

        # Calculate the size category distribution
        size_category_distribution = df_objects['Size Category'].value_counts().sort_index()

        # Calculate class distribution
        class_distribution = df_objects["Label"].value_counts()
        print(class_distribution)
        # dashboard(class_distribution)
        # show_size_distrib(size_category_distribution)
        # show_class_distrib(class_distribution)

        plt.show()

    else :
        for id,f in enumerate(os.listdir(annotations_path)) :
            im_path = os.path.join(annotations_path,f)
            image = Image.open(im_path) 
            # Extract the height and width
            width, height = image.size
            images.append({"Id":id,"Width":width,"Height":height})
            print(f" --> {np.round((id/N)*100,2)} % <--", end="\r")


        # Print images sizes
        df_images  = pd.DataFrame(images)
        print("Images Sizes:")
        print(df_images[["Id", "Width","Height"]])
        df_images.plot(x='Height', y='Width', kind='scatter', title='Images Width vs Height')  
        plt.show()

