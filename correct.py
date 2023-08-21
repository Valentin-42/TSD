import argparse
import yaml
import os 


def correct(label_fld_path, o_path) :
    with open('./configs/nano/data.yaml', 'r') as f:
            table = yaml.safe_load(f)
    table = table['names']
    new_table = {'warning':0,'other':1,'information':2,'regulatory':3,'complementary':4}
    
    for label in os.listdir(label_fld_path) :
        label_path = label_fld_path + "/" + label
        label_path_o = o_path + "/" + label
        with open(label_path, "r") as f : 
            lines = f.readlines()

        with open(label_path_o, "w+") as f : 

            if len(lines) == 0 :
                f.write('')
                continue

            for line in lines :
                for line in lines :
                    l = line.split(' ')
                    new_clc = str(new_table[table[int(l[0])].split('-')[0]])
                    l = f"{new_clc} {l[1]} {l[2]} {l[3]} {l[4]}\n"
                    f.write(l)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./datasets/default_MTSD/train/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="./sliced/", help = "Target folder for a new sliced dataset")
    args = parser.parse_args()
    print("GO")
    if not os.path.exists(args.target):
        os.makedirs(args.target)
    correct(args.source, args. target)