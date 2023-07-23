import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Read a txt file and plot the bounding boxes on the image")
    parser.add_argument("--txt", required=True, help="Path to the txt file")
    parser.add_argument("--img", required=True, help="Path to the image")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Read the txt file
    with open(args.txt, "r") as f:
        lines = f.readlines()

    # Initialize the image
    img = cv2.imread(args.img)
    img_w, img_h, _ = img.shape

    # Loop over the lines in the txt file
    for line in lines:
        # Split the line into the object id, x, y, w, h
        object_id, x, y, w, h = line.split(" ")

        # Convert the x, y, w, h values to floats
        x, y, w, h = float(x)*img_h, float(y)*img_w, float(w)*img_h, float(h)*img_w

        # Draw the bounding box
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Write the object id on the bounding box
        cv2.putText(img, object_id, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the image
    # cv2.imshow("Bounding boxes", img)
    cv2.imwrite("test.jpg",img)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
