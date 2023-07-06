
# This version assume one input of class rating and compare to save only false positives
import sys
import cv2
import matplotlib.pyplot as plt
# Importing Image module from PIL package 
from PIL import Image 
import PIL 
import json,pathlib,os,glob,argparse,sys,xlwt
import os.path as osp
import torch
import torchvision
import torchvision.transforms as T

from xlwt import Workbook
from tqdm import tqdm
from util import get_img_paths, readImage, showImage, get_img_paths
from timm import create_model
# Define transforms for test
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model, load_checkpoint

#default folder
#######################
class_names = ['1','2','3' ,'4' ,'5' ,'6' ,'7', '8' ,'9' ,'10']
class_no = 10
# 10 class
#new model
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'

checkpoint_path_conv_base = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230519-104849-convnext_base_fb_in22k_ft_in1k_384-384/last.pth.tar'
#convnext small
checkpoint_path_conv_small = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230609-150205-convnext_small_fb_in22k_ft_in1k_384-384/last.pth.tar'
checkpoint_path_xcit = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230524-093238-xcit_tiny_24_p16_384_dist-384/last.pth.tar'
checkpoint_path_mobilenetv2 = '/home/pms/pms/pytorch-image-models/output/train/20230616-124910-mobilevitv2_200_384_in22ft1k-384/model_best.pth.tar'
checkpoint_path_xcit_tiny = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230524-093238-xcit_tiny_24_p16_384_dist-384/last.pth.tar'
checkpoint_path_eva2_tiny = '/home/pms/pms/pytorch-image-models/output/train/20230619-130331-eva02_tiny_patch14_336_mim_in22k_ft_in1k-336/model_best.pth.tar'
checkpoint_path_eva2_base = '/media/pms/3da21140-d22d-4c0f-82cb-71cd170f7dbd/home/pms/pytorch-image-models/output/train/20230627-140541-eva02_base_patch14_448_mim_in22k_ft_in1k-448/model_best.pth.tar'
#/media/pms/3da21140-d22d-4c0f-82cb-71cd170f7dbd/home/

input_path = "/home/pms/pms/pms-code/ipsci-script/countLocalRoads/cropImg" 
output_path = "./results/cropImg/52/" #"./results/crop"
detailFile= 'detailed-crop-result.xls'
#######################

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
#SIZE = 366
#SIZE = 384
SIZE = 448


######################################
def get_transform(size,NORMALIZE_MEAN,NORMALIZE_STD):
    if size ==SIZE:
        transforms = [T.Resize((SIZE,SIZE), interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]

        transforms = T.Compose(transforms)
    return transforms
#-----------------

def create_load_model(name,checkpoint_path):
    
    model = create_model(
        name,
        pretrained=False, 
        num_classes=class_no
        )

    load_checkpoint(model,checkpoint_path)
    return model

#-----------------
def saveResult(input_path,img,top1,args):
    if args.savedir:
        out_img_path = osp.join(args.savedir,str(int(class_names[int(str(int(top1.indices[0])))])))
    try:
        os.makedirs(out_img_path , exist_ok=True)
    except OSError as e:
        print(e,"Error: Directory already created or cannot be created. Exiting...")
        exit()
    img_name = osp.basename(input_path)
    #img_name = img_name.split(".")[0]+ img_name.split(".")[1] + "_" + "P_" + str(int(class_names[int(str(int(top1.indices[0])))])) + ".jpg"
    img.save(osp.join(out_img_path,img_name))
######################################

def process(args): 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)   
    if args.model=='swinv2': 
        SIZE = 384 
        model = create_load_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',args.checkpoint)
    elif args.model == 'convnext_base':  
        SIZE = 384  
        model = create_load_model('convnext_base.fb_in22k_ft_in1k_384',args.checkpoint)
    elif args.model == 'convnext_small':  
        SIZE = 384  
        model = create_load_model('convnext_small.fb_in22k_ft_in1k',args.checkpoint)
    elif args.model == 'xcit':
        SIZE = 384
        model =  create_load_model('xcit_tiny_24_p16_384_dist',args.checkpoint)
    elif args.model =='eva2_tiny':
        SIZE = 384
        model =  create_load_model('eva02_tiny_patch16_384.mim_in22k_ft_in1k',args.checkpoint)
    elif args.model =='eva2_base':
        SIZE = 448
        model = create_load_model('eva02_base_patch14_448.mim_in22k_ft_in1k',args.checkpoint)
    elif args.model =='mobilenetv2':
        SIZE = 384
        model = create_load_model('mobilevitv2_200_384_in22ft1k',args.checkpoint)
    else:
        model = create_load_model('swinv2_base_window12to24_192to384_22kft1k',args.checkpoint)

    model.to(device)
    model.eval()
    #transform image
    transforms = get_transform(SIZE,NORMALIZE_MEAN,NORMALIZE_STD);
    print("size = ", SIZE)

    wb2 = Workbook()# Workbook is created
    test2 = wb2.add_sheet('detailed')
    test2.write(0,0,'No')
    test2.write(0,1,'filename')
    test2.write(0,2,'true-class')
    test2.write(0,3,'predicted-class')
    test2.write(0,4,'2nd best-class')
    test2.write(0,5,'3rd best-class')
    test2.write(0,6,'1st probability')
    test2.write(0,7,'2nd probability')
    test2.write(0,8,'3rd probability')
    iter = 1
    count1,count2,count3,count4,count5,count6,count7,count8,count9,count10=0,0,0,0,0,0,0,0,0,0
    paths = get_img_paths(args.path)
    for path in tqdm(paths):
        
        img = PIL.Image.open(path)
        img_tensor = transforms(img).unsqueeze(0).to(device)
        output = torch.softmax(model(img_tensor), dim=1)
        top3 = torch.topk(output, k=3)
        top3_prob = top3.values[0]
        top3_indices = top3.indices[0]
        test2.write(iter, 0, iter)
        test2.write(iter, 1, osp.basename(path))
        test2.write(iter, 2, ' ')
        
        for i in range(3):# uncomment this line if you want to see the best two predictions
            label = class_names[int(str(int(top3_indices[i])))]
            prob = "{:.2f}%".format(float(top3_prob[i])*100)
            test2.write(iter, i+3, label)
            test2.write(iter, i+6, prob)
        
        # write to excel------------------------------------    
        iter = iter + 1 # this iterator is for serial number in excel file to count number of jpg files
                        #-----------------------------------------------------
        top1 = torch.topk(output, k=1)
        if(int(class_names[int(str(int(top1.indices[0])))]) ==10):
            count10=count10+1
        elif(int(class_names[int(str(int(top1.indices[0])))])==9):
            count9=count9+1
        elif(int(class_names[int(str(int(top1.indices[0])))])==8):
            count8=count8+1
        elif(int(class_names[int(str(int(top1.indices[0])))]) ==7):
                count7=count7+1
        elif(int(class_names[int(str(int(top1.indices[0])))]) ==6):
            count6=count6+1
        elif(int(class_names[int(str(int(top1.indices[0])))]) ==5):
            count5=count5+1
        elif(int(class_names[int(str(int(top1.indices[0])))]) ==4):
            count4=count4+1
        elif(int(class_names[int(str(int(top1.indices[0])))]) ==3):
            count3=count3+1
        elif(int(class_names[int(str(int(top1.indices[0])))])==2):
            count2=count2+1
        elif(int(class_names[int(str(int(top1.indices[0])))])==1):
            count1=count1+1
        else:
            print('Error: There is syntax error in the code. Exiting from further rating.')
            break
        #save the image along with its rating
        saveResult(path,img,top1,args)
    print('=========================================================')
    print('Total paths:', len(paths))
    print('one: ',count1,'two: ',count2,'three: ',count3,'four: ',count4,'five: ',count5,'six: ',count6,'seven: ',count7,'eight: ',count8,'nine: ',count9,'ten: ',count10)
    print('=========================================================')
    # save the file
    wb2.save(args.resultFile)

#===============================================

#####################################
def main() :
    parser = argparse.ArgumentParser(description='RoadSurvey inference')
    parser.add_argument("-path", "--path", type=str, help="path to input image Folder", default=input_path)
    parser.add_argument("-checkpoint", "--checkpoint", type=str, help="path to checkpoint file", default=checkpoint_path_swin)
    parser.add_argument('--savedir', type=str, default=output_path, help='The root of save directory')
    parser.add_argument('--resultFile', type=str, default=detailFile, help='The name of xls result File')
    parser.add_argument('--show', action='store_true', help='Whether to show the image',default=False)
    parser.add_argument('--cls', type=int, default=class_no, help='The class rating of the folder')
    parser.add_argument('-model',"--model",type=str,default="swinv2",help="name of the model")
    args = (parser.parse_args())
    if args.savedir:
        #output_path = pathlib.Path(args.savedir)
        try:
            os.makedirs(args.savedir, exist_ok=True)
            print("Directory created")
        except OSError as e:
            print("Error: Directory already exists. Delete the directory and run again.")
            exit()
    if args.model=="convnext_base":
        args.checkpoint = checkpoint_path_conv_base
    elif args.model=="convnext_small":
        args.checkpoint = checkpoint_path_conv_small
    elif args.model=="swinv2" :
        args.checkpoint = checkpoint_path_swin
    elif args.model == 'xcit':
        args.checkpoint = checkpoint_path_xcit
    elif args.model == 'mobilenetv2':
        args.checkpoint = checkpoint_path_mobilenetv2
    elif args.model == 'eva2_tiny':
        args.checkpoint = checkpoint_path_eva2_tiny
    elif args.model == 'eva2_base':
        args.checkpoint = checkpoint_path_eva2_base
    elif args.model =='xcit-tiny':
        args.model = checkpoint_path_xcit_tiny
    else:
        args.checkpoint = checkpoint_path_swin
    process(args)
    print("\nDone!")

#####################################
if __name__ == '__main__':
    main()