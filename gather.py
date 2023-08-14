import argparse
import os
import shutil

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", default="./datasets/tiled/train/labels", help = "Source folder with labels")
    parser.add_argument("-t", default="./datasets/tiled/train/images", help = "Target folder to put corresp images")
    parser.add_argument("-s", default="./datasets/default_MTSD/train/images", help = "Source folder with all images")
    parser.add_argument("-s1", default="./datasets/default_MTSD/val/images", help = "Source folder with all images 2")

    args = parser.parse_args()

    im1 = [f for f in os.listdir(args.s)]
    im2 = [f for f in os.listdir(args.s1)]

    for lbl in os.listdir(args.l) :
        print(lbl)
        corresp = lbl.replace('.txt','.jpg')
        if corresp in im1 :
            im_p = args.s + '/' + corresp
            shutil.copy(im_p, args.t)
        elif corresp in im2 :
            im_p = args.s1 + '/' + corresp
            shutil.copy(im_p, args.t)


