#The code takes an input image and resize to a new size of image with padded zero instead of interpolation
import cv2
import os,sys,glob,argparse
import numpy as np
import tqdm as tqdm
import os.path as osp
from pathlib import Path
from tqdm import tqdm

# Path of the input and output folders
input_folder = '/home/pms/pms-dataset/AcceptedRating2023update/AcceptedRatings/2'
output_folder = '/home/pms/pms-dataset/AcceptedRating2023update/test/2'

#################################
def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        #onlyImages = os.path.join(path, "*" , "*.jpg") # uncomment and comment below if the path has sub directories
        onlyImages = os.path.join(path, "*.jpg") # uncomment if the path do not have sub directories
        paths = sorted(glob.glob(onlyImages))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 
##################################
#-----------------
def saveResult(input_path,img,no=0):

    img_name = osp.basename(input_path)

    if no==1:
        new_img_name = img_name.split(".")[0]+ img_name.split(".")[1] + "_1" + ".jpg"
    elif no==2:
        new_img_name = img_name.split(".")[0]+ img_name.split(".")[1] + "_2" + ".jpg"
    else :
        new_image_name = img_name

    cv2.imwrite(osp.join(output_folder,new_img_name),img)
######################################
def divideImage(img):
    new_img_1 = np.zeros((260, 360, 3), np.uint8)
    new_img_2 = np.zeros((260, 360, 3), np.uint8)    
    if img.shape[:2] == (576, 720): 
        
            # Paste the image into the new image
            new_img_1 = img[300:560,0:360]
            new_img_2 = img[300:560,360:720]
            # Save the new image to the output folder
            return new_img_1,new_img_2
    elif img.shape[:2] == (542,720):
            
            # Paste the image into the new image
            new_img_1 = img[266:526,0:360]
            new_img_2 = img[266:526,360:720]
            # Save the new image to the output folder
            return new_img_1,new_img_2
    elif img.shape[:2] == (558,720):

            # Paste the image into the new image
            new_img_1 = img[282:542,0:360]
            new_img_2 = img[282:542,360:720]
            # Save the new image to the output folder
            return new_img_1,new_img_2
    else:
            print("Image size is wrong: ", img.shape[:2])
            return new_img_1,new_img_2
######################################

def main():
    print("Input Folder: ", input_folder)
# Check if the output folder exists, and create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    paths = get_img_paths(input_folder)
    if not paths:
        print( "Error segmentImage: Did not find any files. Please consult the README." )
        sys.exit(1)
# Loop over all the JPG files in the input folder
    for filename in tqdm(paths):   
        img = cv2.imread(filename)
        # Check if the image has size 700 x 300
        img_1,img_2 = divideImage(img)
        saveResult(filename,img_1,1)
        saveResult(filename,img_2,2)
    print("End Processing")

#####################################
if __name__ == '__main__':
    main()