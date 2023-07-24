import os
import os.path as osp
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

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

def readImage(img_path):
    ori_img = cv2.imread(img_path)
    if ori_img is None:
        return None
    return ori_img

def showImage(image,args):
    if args.show:
        cv2.imshow('view', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit()