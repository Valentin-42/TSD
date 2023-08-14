import os 
import cv2
import argparse
import glob
import json


def im_tiler(im_folder_path,output_im_path, tiling_x, tiling_y ) :

    images_list = [f for f in os.listdir(im_folder_path) if f.endswith(".jpg")]
    for index,im_name in enumerate(images_list) :
        print(f"image : {index} | {im_name}")
        im = cv2.imread(os.path.join(im_folder_path,im_name))
        o_im = im.copy()

        im_height,im_width, _ = o_im.shape
        print(f"{ im_width} , {im_height}")

        for i in range (0,im_width,int(tiling_x)) :
            for j in range(0,im_height,int(tiling_y)) :
                # Extract a tile from the image
                tile = o_im[j:j+tiling_y, i:i+tiling_x]
                
                # Construct a tile name based on its position
                tile_name = f"tile_{i}_{j}_{im_name}"

                # Save the tile in a specified directory
                tile_save_path = os.path.join(output_im_path, tile_name)
                cv2.imwrite(tile_save_path, tile)
                
    print("Tiling completed.")

def an_tiler(an_folder_path,output_path,tiling_x,tiling_y) :

    an_list = [f for f in os.listdir(an_folder_path) if f.endswith(".json")]
    for an in an_list :
        print(an)
        with open(os.path.join(an_folder_path,an), 'r') as json_file:
            data = json.load(json_file)
            im_width = data['width']
            im_height = data['height']
            objects = data['objects']

            print(f" w : {im_width} , h : {im_height}")

            for i in range(0, im_width, int(tiling_x)):
                for j in range(0, im_height, int(tiling_y)):
                    new_objects = []
                    
                    for obj in objects:
                        bbox=obj["bbox"]
                        x = bbox['xmin']
                        y = bbox['ymin']
                        width = bbox['xmax'] - bbox['xmin']
                        height = bbox['ymax'] - bbox['ymin']
                        
                        # Check if the object's bounding box entirely insisde the current tile
                        if i <= x + width <= i + tiling_x and j <= y + height <= j + tiling_y and i <= x <= i + tiling_x and j <= y <= j + tiling_y:
                            # Calculate normalized coordinates within the tile
                            norm_x = (x - i) / tiling_x
                            norm_y = (y - j) / tiling_y
                            norm_width = width / tiling_x
                            norm_height = height / tiling_y

                            new_objects.append({
                                'label': obj['label'],
                                'x': norm_x,
                                'y': norm_y,
                                'width': norm_width,
                                'height': norm_height
                            })
                        # Check if the object's bounding box top left insisde the current tile
                        elif i <= x <= i + tiling_x and j <= y <= j + tiling_y :
                            # Calculate normalized coordinates within the tile
                            norm_x = (x - i) / tiling_x
                            norm_y = (y - j) / tiling_y
                            norm_width  = (min((x - i) + width,tiling_x) -  (x - i) )/ tiling_x
                            norm_height = (min((y - j) + height,tiling_y) - (y - j) )/ tiling_y

                            new_objects.append({
                                'label': obj['label'],
                                'x': norm_x,
                                'y': norm_y,
                                'width': norm_width,
                                'height': norm_height
                            })
                        
                        elif i <= x + width <= i + tiling_x and j <= y + height <= j + tiling_y :
                            # Calculate normalized coordinates within the tile
                            norm_x = max(0,(x - i)) / tiling_x
                            norm_y = max(0,(y - j)) / tiling_y
                            norm_width  = ((x + width - i) - max(0,(x - i)) )/ tiling_x
                            norm_height = ((y + height -j) - max(0,(y - j)) )/ tiling_y

                            new_objects.append({
                                'label': obj['label'],
                                'x': norm_x,
                                'y': norm_y,
                                'width': norm_width,
                                'height': norm_height
                            })
                        
                        elif i <= x <= i + tiling_x and j <= y + height <= j + tiling_y :
                            # Calculate normalized coordinates within the tile
                            norm_x = max(0,(x - i)) / tiling_x
                            norm_y = max(0,(y - j)) / tiling_y
                            norm_width  = ((x + width - i) - max(0,(x - i)) )/ tiling_x
                            norm_height = ((y + height -j) - max(0,(y - j)) )/ tiling_y

                            new_objects.append({
                                'label': obj['label'],
                                'x': norm_x,
                                'y': norm_y,
                                'width': norm_width,
                                'height': norm_height
                            })
                        elif i <= x + width <= i + tiling_x and j <= y <= j + tiling_y :
                            # Calculate normalized coordinates within the tile
                            norm_x = max(0,(x - i)) / tiling_x
                            norm_y = min(max(0,(y - j)),tiling_y) / tiling_y
                            norm_width  = ((x + width  - i) - max(0,(x - i)) )/ tiling_x
                            norm_height = ((y + height - j) - max(0,(y - j)) )/ tiling_y

                            new_objects.append({
                                'label': obj['label'],
                                'x': norm_x,
                                'y': norm_y,
                                'width': norm_width,
                                'height': norm_height
                            })

                    txt_name = an.split(".")[0]+".txt"
                    tile_name = f"tile_{i}_{j}_{txt_name}"
                    with open(os.path.join(output_path, tile_name),"w+") as txt_file:
                        if not len(new_objects) > 0 :
                            txt_file.write("")
                        print(len(new_objects))
                        for obj in new_objects :
                            f = f"{obj['label']} {obj['x']} {obj['y']} {obj['width']} {obj['height']} \n"
                            txt_file.write(f)

                    

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-sourcei", default="./", help = "Source folder with images")
    parser.add_argument("-sourcean", default="./", help = "Source folder with labels")
    parser.add_argument("-target", default="./yolosliced/ts/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".jpg", help = "Image extension in a dataset. Default: .jpg")
    parser.add_argument("-sizex", type=int, default=256, help = "Size of a tile. Default: 256")
    parser.add_argument("-sizey", type=int, default=256, help = "Size of a tile. Default: 256")

    args = parser.parse_args()

    im_path = args.sourcei
    lab_path = args.sourcean
    
    if len(im_path) == 0:
        raise Exception("Source folder should contain some images")

    if len(lab_path) == 0:
        raise Exception("Source folder should contain some images")

    if not os.path.exists(args.target):
        os.makedirs(args.target)
    elif len(os.listdir(args.target)) > 0:
        raise Exception("Target folder should be empty")
    
    print(f"Starting tiling images in {int(args.sizex)} x {int(args.sizey)}")
    im_tiler(im_path,args.target,args.sizex, args.sizey)
    an_tiler(lab_path,args.target,args.sizex, args.sizey)
    
