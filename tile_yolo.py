import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import argparse
import os
import random
import shutil
import json
import yaml

def tiler(imnames, newpath, slice_size, ext):

    for index,imname in enumerate(imnames):
        print(f"{round(100*index/len(imnames))} %",end="\r")
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        width, height = im.size

        data = {'width':width, 'height':height, 'objects':[]}
        lbl = imname.replace(ext, '.txt').replace('images', 'labels')
        with open(lbl, 'r') as f:
            lines = f.readlines()
        for line in lines :
            line.split(' ')
            if len(line) > 0 :
                data['objects'].append(line.split(' '))

        objects = data['objects']
        width = data['width']
        height = data['height']
        boxes = []
        for obj in objects:
            x = float(obj[1]) * width
            y = float(obj[2]) * height
            bwidth =float(obj[3]) * width
            bheight = float(obj[4]) * height

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bwidth)
            y2 = int(y + bheight)

            boxes.append((obj[0], Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
      
        counter = 0
        # create tiles and find intersection with bounding boxes for each tile
        for i in range(0, width, slice_size):
            for j in range(0, height, slice_size):
                x1 = i
                y1 = j
                x2 = (i + slice_size) - 1
                y2 = (j + slice_size) - 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])
                        
                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy
                        
                        # get bounding box width and height normalized to slice size
                        new_width =  (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size

                        # we have to normalize central x and invert y for yolo format
                        new_x = ( min(x) - i  ) / slice_size
                        new_y = ( min(y) - j) / slice_size

                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])

                    if not imsaved:
                        sliced = imr[j:j+slice_size, i:i+slice_size]
                        sliced_im = Image.fromarray(sliced)
                        filename = imname.split('/')[-1]
                        slice_path = newpath+"/"+filename.replace(ext, f'_{i}_{j}{ext}')                      
                        slice_labels_path = newpath+"/"+filename.replace(ext, f'_{i}_{j}.txt')                      
                        sliced_im.save(slice_path)
                        imsaved = True
            
                if len(slice_labels) > 0:
                    with open(slice_labels_path,"w+") as txt_file:
                        for obj in slice_labels :
                            f = f"{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]} \n"
                            txt_file.write(f)
                else :
                    with open(slice_labels_path, "w+") as f :
                        f.write("")

def filter(label_path, cnt, max_per_clc) : 

    with open('./configs/nano/data.yaml', 'r') as f:
        table = yaml.safe_load(f)
    table = table['names']
    for k in table.keys() :
        table[k] = table[k].split('-')[0]

    new_table = {'warning':0,'other':1,'information':2,'regulatory':3,'complementary':4}

    # FILTER        
    with open(label_path, "r") as f : 
        lines = f.readlines()

    if len(lines) == 0 :
        if cnt['empty'] > max_per_clc :
                return cnt, True
        cnt['empty']+=1
        return cnt, False
        
    for line in lines :
        if line == "" or line == " " :
            if cnt['empty'] > max_per_clc :
                return cnt, True
            cnt['empty']+=1
        else :
            clc = line.split(' ')[0]
            clc = new_table[table[clc]]
            if cnt[clc] > max_per_clc :
                return cnt, True
            cnt[clc] +=1

    return cnt, False

def splitter(source, target, ext, ratio, max):
    labnames = [source+"/"+f.replace(ext, '.txt') for f in os.listdir(source)]
    if len(labnames) == 0 :
        print("error", source)
        return
    elif max > len(labnames) :
        print(" /!\ Dataset size asked superior to available images => taking max possible ")



    t_train_im = os.path.join(target, 'train/images')
    t_train_lab = os.path.join(target, 'train/labels')

    t_val_im = os.path.join(target, 'val/images')
    t_val_lab = os.path.join(target, 'val/labels')

    # split dataset for train and test
    max_train = int(max * ratio)
    max_val   = int(max * (1-ratio))

    cnt_train = {'total':0, 'empty':0, '0':0, '1':0, '2':0, '3':0, '4':0}
    cnt_val   = {'total':0, 'empty':0, '0':0, '1':0, '2':0, '3':0, '4':0}
    i = 0
    print("Starting ... ")
    for name in labnames:

        im_path     = name.replace('.txt', '.jpg')                       
        labels_path = name
        if random.random() > ratio:
            cnt_val, jump = filter(labels_path, cnt_val, max_val//6)
            if jump : 
                continue
            shutil.copy(im_path    , t_val_im)
            shutil.copy(labels_path, t_val_lab)
            cnt_val['total']+=1

        else:
            cnt_train, jump = filter(labels_path, cnt_train, max_train//6)
            if jump :
                continue
            shutil.copy(im_path    , t_train_im)
            shutil.copy(labels_path, t_train_lab)
            cnt_train['total']+=1


        i+=1
        if i == max :
            break

    print('== Dataset created ==') 
    print('train:', cnt_train)
    print('val:', cnt_val)
         
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./datasets/default_MTSD/train/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="./sliced/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".jpg", help = "Image extension in a dataset. Default: .jpg")
    parser.add_argument("-size", type=int, default=640, help = "Size of a tile. Default: 512")
    parser.add_argument("-split", type=bool, default=True, help = "True : Split into dataset")
    parser.add_argument("-max", type=int, default=5000, help = "Number of total images")
    parser.add_argument("-ratio", type=float, default=0.8, help = "Train/val split ratio from max. Dafault: 0.8")
    parser.add_argument("-keep", type=bool, default=False, help = "Keep cache folder")

    args = parser.parse_args()

    labnames = [args.source+"/labels/"+f for f in os.listdir(args.source+"/labels/") if f.endswith(".txt")]
    imnames  = [f.replace('txt', 'jpg').replace('labels','images') for f in labnames]
    
    print(f"{len(labnames)} , {len(imnames)}")
    
    if len(imnames) == 0:
        raise Exception(f"Source folder should contain some images {args.source}")
    elif len(imnames) != len(labnames):
        print(args.source+"/labels/")
        print(args.source+"/images/")
        raise Exception(f"Dataset should contain equal number of images and txt files with labels {len(labnames)} != {len(imnames)}")


    if not os.path.exists(args.target):
        os.makedirs(args.target)
        os.makedirs(os.path.join(args.target,'train','images'))
        os.makedirs(os.path.join(args.target,'train','labels'))
        os.makedirs(os.path.join(args.target,'val','images'))
        os.makedirs(os.path.join(args.target,'val','labels'))
        os.makedirs(os.path.join(args.target,'cache'))
    elif len(os.listdir(args.target)) > 0:
        raise Exception("Target folder should be empty")

    print(f"== Start Tiling {args.max} img== ")
    imnames = [f for i,f in enumerate(imnames) if i<args.max]
    tiler(imnames, os.path.join(args.target,'cache'), args.size, args.ext)
    if args.split :
        splitter(os.path.join(args.target,'cache'), args.target, args.ext, args.ratio, args.max)

    if not args.keep :
        shutil.rmtree(os.path.join(args.target,'cache'))