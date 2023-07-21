from __future__ import print_function, absolute_import, division
import numpy as np
import cv2
import os
import os.path as osp
import glob

from pathlib import Path
from tqdm import tqdm
from util import get_palette, get_classes
from util import get_img_paths, readImage, showImage, get_img_paths
#from mmseg.apis import init_model, inference_model, show_result_pyplot,inference_segmentor
from mmseg.apis import inference_model, init_model, show_result_pyplot
import argparse, os, glob, sys,random, json, base64,io

#default folder
input_path = "/home/pms/pms-dataset/RMOGalwayCountLocalRoads/images"
output_path = "./countLocalRoads"
segImg_path = "segImg/"
cropImg_path = "cropImg/"


#####################################
def result_segImage(result,
                    palette=None,
                    CLASS_NAMES=None):
        """compute`resulted mask` for the `img`.

        """
        seg = result.cpu()
        if palette is None:
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3))
            np.random.set_state(state)

        palette = np.array(palette)
        assert palette.shape[0] == len(CLASS_NAMES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if(label == 0):
                continue
            if(label == 1):
                continue
            if(label == 2):
                continue
            if(label == 3): # road
                color_seg[seg == label, :] = [255,255,255] #color
            if(label == 4): # 
                continue
            if(label == 5): #
                continue
            if(label == 6): #
                continue
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        return color_seg
#####################################
def save(segImg_path,cropImg_path,path,args,segImage,orig):
    path = path.replace(".JPG",".jpg")
    if args.savedir:
        seg_out = os.path.join(args.savedir,segImg_path)
        crop_out = os.path.join(args.savedir,cropImg_path)
    try:
        os.makedirs(seg_out, exist_ok=True)
        os.makedirs(crop_out, exist_ok=True)
    except OSError as e:
        print(e,"Warning: Directory already created or cannot be created")
        
    cv2.imwrite(os.path.join(seg_out,osp.basename(path)),segImage)
    cv2.imwrite(os.path.join(crop_out,osp.basename(path)),orig)

######################################
def segmentImage(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint , device='cuda:0')
    #model = init_segmentor(args.config, args.checkpoint  , device='cpu') # this does not work
    # inference the model on a given image
    paths = get_img_paths(args.path)
    ITER = 0;
    if not paths:
        print( "Error segmentImage: Did not find any files. Please consult the README." )
        sys.exit(1)
    for path in tqdm(paths):
        image = cv2.imread(path) # Read the image using opencv format

        if image.shape[:2] == (576, 720):
            orig = np.zeros((576, 720, 3), np.uint8)
            orig = image
        elif image.shape[:2] == (542,720):
            orig = np.zeros((576, 720, 3), np.uint8)
            orig [34:576,0:720] = image
        elif image.shape[:2] == (558,720):
            orig = np.zeros((576, 720, 3), np.uint8)
            orig [18:576,0:720] = image
        else:
            orig = cv2.resize(image, (576, 720)) #(h,w)
            
        #result = inference_segmentor(model, orig)
        result = inference_model(model,orig)
        #sem_seg = sem_seg.cpu().data
        result2 = (result.pred_sem_seg.data[0])
       
        mask = result_segImage(result2, palette=get_palette('roadsurvey'), CLASS_NAMES=get_classes('roadsurvey'))
        segImage = cv2.bitwise_and(orig, mask)
        #following is the code to automatically discard images with poor segmentation result
        gray_mask = cv2. cvtColor(mask, cv2.COLOR_BGR2GRAY)
        count = cv2.countNonZero(gray_mask)
        contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thres = (orig.shape[0]*orig.shape[1])*(0.20) # a heruristic value
        if count<thres or len(contours)>20: # a heuristic value
            #showImage(segImage,args)
            ITER = ITER +1
            print(path, " :count: ", ITER)
            continue
        # end of image removal code
        else:
            save(segImg_path,cropImg_path,path,args,segImage[230:560,0:700],orig[230:560,0:700])
            #a = []
            #a.append(segImage[230:560,0:700])
            #a.append(orig[230:560,0:700])
            #for image in a:
            #    showImage(image,args)
            #exit()
    cv2.destroyAllWindows()

#####################################
    

#####################################
def main() :
    parser = argparse.ArgumentParser(description='RoadSurvey inference')
    parser.add_argument("-path", "--path", type=str, help="path to input image Folder", default=input_path)
    parser.add_argument("-config", "--config", type=str, help="path to config file", default="/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py")
    parser.add_argument("-checkpoint", "--checkpoint", type=str, help="path to checkpoint file", default="/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth")
    parser.add_argument('--savedir', type=str, default=output_path, help='The root of save directory')
    parser.add_argument('--show', action='store_true', help='Whether to show the image',default=False)
    args = (parser.parse_args())

    path = os.path.join(os.getcwd(), "{}".format(args.path))
    config = os.path.join(os.getcwd(), "{}".format(args.config))
    checkpoint = os.path.join(os.getcwd(), "{}".format(args.checkpoint))
    roadsurveyPath = path
    print("roadsurveyPath: ", parser, "and", args)
    if args.savedir:
        #output_path = pathlib.Path(args.savedir)
        try:
            os.makedirs(args.savedir, exist_ok=True)
            print("Directory created")
        except OSError as e:
            print("Error: Directory already exists. Delete the directory and run again.")
            exit()
    
    segmentImage(args)
    print("\nDone!")


#####################################
if __name__ == '__main__':
    main()

