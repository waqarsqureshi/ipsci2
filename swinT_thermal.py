import argparse
import cv2
import numpy as np
from PIL import Image 
import PIL 
from tqdm import tqdm
import sys,os
import os.path as osp
from pathlib import Path
import torch
import timm
from timm.models import create_model, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from util import get_img_paths, readImage, showImage, get_img_paths
class_names = ['1','2','3' ,'4' ,'5' ,'6' ,'7', '8' ,'9' ,'10']
class_no = 10
NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
#SIZE = 366
SIZE = 384
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def create_load_model(name,checkpoint_path):
    
    model = create_model(
        name,
        pretrained=False, 
        num_classes=class_no
        )

    load_checkpoint(model,checkpoint_path)
    return model

#####################################

def saveThermal(path,outputImg,savedir):
    path = path.replace(".JPG",".jpg")
    try:
        os.makedirs(savedir, exist_ok=True)
    except OSError as e:
        print(e,"Warning: Directory already created or cannot be created")
        
    cv2.imwrite(os.path.join(savedir,osp.basename(path)),outputImg)

######################################
def get_transform(size,NORMALIZE_MEAN,NORMALIZE_STD):
    if size ==SIZE:
        transforms = [T.Resize((SIZE,SIZE), interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]

        transforms = T.Compose(transforms)
        return transforms
    return 1
#-----------------



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--input_path',
        type=str,
        default='./2.jpg',
        help='Input image path')
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='./thermal/',
        help='Input image path')
    
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='xgradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = create_load_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',checkpoint_path_swin)
    model.eval()
   
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]
    #targets = [ClassifierOutputTarget(3)] #uncomment for specific class
    targets = None #uncomment for the top prediction
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    paths = get_img_paths(args.input_path) # assume the path contains images
    if not paths:
        print( "Error segmentImage: Did not find any files. Please consult the README." )
        sys.exit(1)
    for path in tqdm(paths):    
        rgb_img = cv2.imread(path, 1)[:, :, ::-1]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (384, 384))
        rgb_img = np.float32(rgb_img) / 255 # this is rgb_img_float
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# We have to specify the target we want to generate the Class Activation Maps for.
# If targets is None, the highest scoring category will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=args.eigen_smooth,aug_smooth=args.aug_smooth)
        grayscale_cam = grayscale_cam[0, :]
        #gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        #gb = gb_model(input_tensor, target_category=None)
        #gb = deprocess_image(gb)
        #cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        #cam_gb = deprocess_image(cam_mask * gb)
        #cam_gb = cv2.cvtColor(gb,cv2.COLOR_RGB2GRAY)
        #cam_gb = np.float32(gb) / 255 
        cam_image = show_cam_on_image(rgb_img,grayscale_cam, use_rgb=False,image_weight=0.8,colormap=cv2.COLORMAP_HOT)
        cam_image = cv2.resize(cam_image  ,(700, 330)) 
        #saveThermal(path,gb_image ,args.output_path)
        saveThermal(path,cam_image,args.output_path)