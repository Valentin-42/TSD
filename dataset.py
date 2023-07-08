import json
import os
import shutil

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

if __name__ == '__main__':
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


