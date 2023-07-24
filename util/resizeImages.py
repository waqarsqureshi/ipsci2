#The code takes an input image and resize to a new size of image with padded zero instead of interpolation
import cv2
import os
import numpy as np
import tqdm as tqdm

# Path of the input and output folders
input_folder = '/home/pms/pms-dataset/AcceptedRating2023update/iPSCI/2'
output_folder = '/home/pms/pms-dataset/AcceptedRating2023update/iPSCI-resized/2'
print("Input Folder: ", input_folder)
# Check if the output folder exists, and create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Loop over all the JPG files in the input folder
for filename in os.listdir(input_folder):
    
    if filename.endswith('.jpg'):
        # Load the image
        #print("Processing:", filename)
        img = cv2.imread(os.path.join(input_folder, filename))
        # Check if the image has size 700 x 300
        if img.shape[:2] == (330, 700):
            # Create a new 720 x 576 image
            new_img = np.zeros((576, 720, 3), np.uint8)

            # Paste the image into the new image
            new_img[0:330, 10:710] = img

            # Save the new image to the output folder
            cv2.imwrite(os.path.join(output_folder, filename), new_img)
        elif img.shape[:2] == (272, 700):
            # Create a new 720 x 576 image
            new_img = np.zeros((576, 720, 3), np.uint8)

            # Paste the image into the new image
            new_img[0:272, 10:710] = img

            # Save the new image to the output folder
            cv2.imwrite(os.path.join(output_folder, filename), new_img)
        elif img.shape[:2] == (288, 700):
            # Create a new 720 x 576 image
            new_img = np.zeros((576, 720, 3), np.uint8)

            # Paste the image into the new image
            new_img[0:288, 10:710] = img

            # Save the new image to the output folder
            cv2.imwrite(os.path.join(output_folder, filename), new_img)
        else:
            print("Image size is wrong: ", img.shape[:2])
    else:
        print("No Image Files:...")
print("End Processing")