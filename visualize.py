import argparse
import cv2
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Read a txt file and plot the bounding boxes on the image")
    parser.add_argument("--txt", default="./sliced/cache/",required=False, help="Path to the txt file")
    parser.add_argument("--img", default="./sliced/cache/", required=False, help="Path to the image")
    parser.add_argument("--fld", default=True, required=False, help="Is a folder ?")
    parser.add_argument("--save_fld", default="./sliced/cache/", required=False, help="Target Folder")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if bool(args.fld) == True :
        l = [f for f in os.listdir(args.img) if f.endswith(".jpg")]
        for img in l :
            name = img.split(".jpg")[0]
            an   = os.path.join(args.txt,name+".txt")
            im   = os.path.join(args.img,img)
            draw(an,im,args.save_fld) 
    else :
        draw(args.txt, args.img. args.save_fld)

def draw(text_file, image_file, output_path) :

    # Read the txt file
    with open(text_file, "r") as f:
        lines = f.readlines()

    # Initialize the image
    img = cv2.imread(image_file)
    img_h, img_w, _ = img.shape
    # Loop over the lines in the txt file
    for line in lines:
        if line == " " : 
            continue
        
        print(image_file)

        # Split the line into the object id, x, y, w, h
        object_id, x, y, w, h, _= line.split(" ")

        # Convert the x, y, w, h values to floats
        x, y, w, h = float(x)*img_h, float(y)*img_w, float(w)*img_h, float(h)*img_w

        print(f"{x}, {y}, {w}, {h}")

        # Draw the bounding box
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Write the object id on the bounding box
        cv2.putText(img, object_id, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the image
    # cv2.imshow("Bounding boxes", img)
    cv2.imwrite(os.path.join(output_path,image_file.split("/")[-1].split(".")[0]+"_vis.jpg"),img)
    # cv2.waitKey(0)



if __name__ == "__main__":
    main()
