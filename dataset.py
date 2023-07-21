import json
import os
import shutil
import random

labels = []


def Create_Dataset_Architecture(ds_root,dataset_name) :
    folders    = ["train", "val", "test"]
    subfolders = ["images","labels"]

    os.mkdir(ds_root+dataset_name)
    root = ds_root+dataset_name+"/"
    for folder in folders :
        os.mkdir(root+folder)
        for subfolder in subfolders :
            os.mkdir(root+folder+"/"+subfolder)

    return root+"/train/",root+"/val/",root+"/test/"

def move_annotation(image_key,folder):
    path = os.path.join('./MTSD/extracted/mtsd_v2_fully_annotated/annotations/', '{:s}.json'.format(image_key))
    shutil.copy(path, folder+"/labels/")

def move_image(image_key,folder):
    path = os.path.join('./MTSD/extracted/images/', '{:s}.jpg'.format(image_key))
    shutil.copy(path, folder+"/images/")

def json_to_COCO_format(json_file_path, single_class=False) :

    # output : <object-class-id> <x> <y> <width> <height>
    txt_file_path = os.path.splitext(json_file_path)[0] + ".txt"
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        w = data['width']
        h = data['height']
        objects = data['objects']

        with open(txt_file_path, 'w') as text_file:
            for obj in objects:
                label = obj['label']
                if single_class :
                    label_id = "0"
                else :
                    if label in labels :
                        label_id = labels.index(label)
                    else :
                        labels.append(label)
                        label_id = labels.index(label)
                        print(f"New class {label} : id {len(labels)}")
                bbox  = obj['bbox']
                xmin  = bbox['xmin']
                ymin  = bbox['ymin']
                xmax  = bbox['xmax']
                ymax  = bbox['ymax']
                
                width_n = (xmax - xmin)/w
                height_n = (ymax - ymin)/h
                
                line = f"{label_id} {xmin/w} {ymin/h} {width_n} {height_n}\n"
                text_file.write(line)

def normalize_txt():
    import json
    # <object-class-id> <x> <y> <width> <height> 
    train_label_path = "./datasets/light_MTSD/train/labels/"
    train_image_path = "./datasets/light_MTSD/train/images/"

    for file in os.listdir(train_label_path):
        print(file)
        txt_file_path = train_label_path + file
        im_file_path = train_image_path + os.path.splitext(file)[0] + ".jpg"

        # Open im_file_path and get image width and height 
        with open(im_file_path, 'rb') as image_file:
            from PIL import Image
            image = Image.open(image_file)
            im_w, im_h = image.size

        with open(txt_file_path, 'r') as txt_file:
            lines = txt_file.readlines()

        # Process each line and save the modified txt file
        modified_lines = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 5:
                if (x<1 or y<1 or width<1 or  height<1) :
                    continue 
                object_class_id = line[0]
                x = float(line[1]) / im_w
                y = float(line[2]) / im_h
                width = float(line[3]) / im_w
                height = float(line[4]) / im_h
                
                modified_line = f"{object_class_id} {x} {y} {width} {height}\n"
                modified_lines.append(modified_line)

        # Save the modified txt file
        modified_txt_file_path = txt_file_path
        with open(modified_txt_file_path, 'w') as modified_txt_file:
            modified_txt_file.writelines(modified_lines)      

def dataset_creation_based_on_splits_file() :
    print("Creating dataset")

    path_to_datasets = "./datasets/"
    path_to_splits_folder = "./MTSD/extracted/mtsd_v2_fully_annotated/splits/"
    file_train = path_to_splits_folder + "train.txt"
    file_val   = path_to_splits_folder + "val.txt"
    file_test  = path_to_splits_folder + "test.txt"
    files = [file_train, file_val, file_test]

    dataset_name = "MTSD"
    print("Init Ok")
    train_path, val_path, test_path = Create_Dataset_Architecture(path_to_datasets,dataset_name)
    print("MTSD ready")

    for i,f in enumerate(files) : 
        content = open(f, 'r')
        Lines = content.readlines()
        print(i)
        for j,key in enumerate(Lines):
            key =  key.replace("\n", "")
            if   i == 0 :
                move_annotation(key,train_path)
                move_image(key,train_path)
                print(f"train : {j}",end="\r")
            elif i == 1 :
                move_annotation(key,val_path)
                move_image(key,val_path)
                print(f"val : {j}",end="\r")
            elif i == 2 :
                # move_annotation(key,test_path)
                move_image(key,test_path)
                print(f"test : {j}",end="\r")
        print("Done")

    print("DONE")

def create_light_dataset(folder,factor):

    folder_path = "./datasets/default_MTSD/"+folder+'/'
    im_path = "./datasets/default_MTSD/"+folder+"/images/"
    an_path = "./datasets/default_MTSD/"+folder+"/labels/"

    out_im_path = "./datasets/light_MTSD/"+folder+"/images/"
    out_an_path = "./datasets/light_MTSD/"+folder+"/labels/"

    # Get all file names in the folder
    file_names = os.listdir(im_path)

    # Calculate the number of files to remove
    num_files_to_extract = len(file_names) // factor

    # Randomly select files
    selected_files = random.sample(file_names, num_files_to_extract)
    for i,file_name in enumerate(selected_files):
        an_file_name = file_name.split(".")[0] + ".txt"
        image_path    = os.path.join(im_path, file_name)
        annot_path    = os.path.join(an_path, an_file_name)

        dest_im_path   =  os.path.join(out_im_path, file_name)
        dest_an_path   =  os.path.join(out_an_path, an_file_name)

        shutil.copy(image_path,dest_im_path)
        if not folder == "test" :
            shutil.copy(annot_path,dest_an_path)
        print(f".. {file_name} .. -> {100*(i/num_files_to_extract)} %")




if __name__ == '__main__':

    path_to_datasets = "./datasets/"
    path_to_ds = path_to_datasets + "default_MTSD/"
    path_to_train_labels = path_to_ds + "train/labels/"
    path_to_val_labels   = path_to_ds + "val/labels/"
    labels_folders = [path_to_train_labels,path_to_val_labels]

    # 1. Create COCO txt labels from json
    for folder in labels_folders :
        for json_f in os.listdir(folder) :
            print(folder+json_f)
            if not json_f.endswith(".json") :
                continue
            json_to_COCO_format(json_file_path=folder+json_f,single_class=False)

    txt_file_path = "classes_desc.txt"
    with open(txt_file_path, 'w') as text_file:
        for label in labels:
            label_id = labels.index(label)
            line = f"{label_id}: '{label}' \n"
            text_file.write(line)

    # Create_Dataset_Architecture(path_to_datasets,"light_MTSD")
    # create_light_dataset("test",4)

    # normalize_txt()

    # train_image_path = "./datasets/light_MTSD/train/images/"
    # train_label_path = "./datasets/light_MTSD/train/labels/"

    # for image in os.listdir(train_image_path) :
    #     print(image.split('.')[0])
    #     move_annotation(image.split('.')[0],train_label_path)


    # for ann in os.listdir(train_label_path) :
    #     ann = train_label_path+ann
    #     json_to_COCO_format(ann, single_class=True)