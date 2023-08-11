import argparse,os
import cv2
import numpy as np
import torch
import timm
from tqdm import tqdm
import os.path as osp
from pathlib import Path
from timm.models import create_model, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from PIL import Image 
import PIL 
from pathlib import Path
from util import get_img_paths, readImage, showImage, get_img_paths
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image, create_labels_legend
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.ablation_layer import AblationLayerVit

from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
#=========================================================================
class_no = 10
class_names = ['1','2','3','4','5','6','7','8','9','10']
class_dictionary = {index: label for index, label in enumerate(class_names)}
#SIZE = 366
SIZE = 384
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
#==========================================================================
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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input_path',type=str,default='./2.jpg',help='Input image path')
    parser.add_argument('--output_path',type=str,default='./thermal/',help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',action='store_true',help='Reduce noise by taking the first principle componenetof cam_weights*activations')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    device = 'cuda'
    if args.use_cuda:
        print('Using GPU for acceleration')
        device = 'cuda'
    else:
        print('Using CPU for computation')
        device = 'cpu'
    return args
#=======================================================
def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
#=======================================================
if __name__ == '__main__':
    """ python vizualizeClassifierSwinV2.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    
    """
    args = get_args()
    #model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = create_load_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',checkpoint_path_swin)
    model.eval()
    paths = get_img_paths(args.input_path) # assume the path contains images
######################################
    
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]
    #targets = [ClassifierOutputTarget(3)] #uncomment for specific class
    targets = None #uncomment for the top prediction

    if not paths:
        print( "Error segmentImage: Did not find any files. Please consult the README." )
        sys.exit(1)
    
    for path in tqdm(paths):    
        rgb_img = cv2.imread(path, 1)[:, :, ::-1]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (384, 384))
        rgb_img_float = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        dff = DeepFeatureFactorization(model=model, target_layer=target_layers, reshape_transform=reshape_transform,computation_on_concepts=model.head.fc)
        concepts, batch_explanations, concept_scores = dff(input_tensor, n_components=5) # number of components to be displayed
        concept_outputs = torch.softmax(torch.from_numpy(concept_scores), axis=-1).numpy()
        concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :2] # just to get a sorted list for debugging
        concept_label_strings = create_labels_legend(concept_outputs, class_dictionary,top_k=1) #this top_k is sub category so ignore here
        visualization = show_factorization_on_image(rgb_img_float, batch_explanations[0],image_weight=0.8,concept_labels=concept_label_strings)
        cam_image = cv2.resize(visualization,(700, 330))
        saveThermal(path,cam_image,args.output_path) 