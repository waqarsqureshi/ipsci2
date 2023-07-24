import splitfolders
import sys
import argparse
import os
argparse = argparse.ArgumentParser()

argparse.add_argument(
    "-i", "--inputPath", type=str, help="path to input directory of images"
)

argparse.add_argument(
    "-o", "--outputPath", type=str, help="path to output directory of images"
)

args = vars(argparse.parse_args())
if(args["inputPath"] == None):
    print("ERROR: Please provide input path and output path e.g. python3 splitFolders.py -i ./input_directory -o ./output_directory")
    sys.exit()
if(args["outputPath"] == None):
    print("ERROR: Please provide input path and output path, e.g. python3 splitFolders.py -i ./input_directory -o ./output_directory")
    sys.exit()
else:
    input_path = os.path.join(os.getcwd(), "{}".format(args["inputPath"]))
    output_path = os.path.join(os.getcwd(), "{}".format(args["outputPath"]))
    try:
        os.makedirs(output_path , exist_ok=True)
    except:
        print("Directory already exists")
    print("INFO: Input path: {}".format(args["inputPath"]))
    print("INFO: Output path: {}".format(args["outputPath"]))

#input and output test paths

#input_path = "/home/pms/pms-dataset/pms/pms-dataset/dataset/pavement"
#output_path="/home/pms/pms-dataset/pms/pms-dataset/dataset/pavement/output"

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_path, output=output_path,seed=1337, ratio=(.70, .30), group_prefix=None, move=False) # default values
print("INFO: Splitting complete")