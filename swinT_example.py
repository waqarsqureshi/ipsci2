import argparse,os
import cv2
import numpy as np
import torch
import timm
from timm.models import create_model, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from PIL import Image 
import PIL 
from pathlib import Path
from util import get_img_paths, readImage, showImage, get_img_paths
class_no = 10
class_names = ['1','2','3','4','5','6','7','8','9','10']
class_dictionary = {index: label for index, label in enumerate(class_names)}
#print(class_dictionary)

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
device = 'cuda'
#SIZE = 366
SIZE = 384
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
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


def create_load_model(name,checkpoint_path):
    
    model = create_model(
        name,
        pretrained=False, 
        num_classes=class_no
        )

    load_checkpoint(model,checkpoint_path)
    return model


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
        '--image_path',
        type=str,
        default='./1.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory to save the images')
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
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
    print("---------------------------------------------")
    print(model.head.fc)

    transform = get_transform(SIZE,NORMALIZE_MEAN,NORMALIZE_STD);
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

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

    #rgb_img = PIL.Image.open(args.image_path)
    #input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (384, 384))
    rgb_img_float = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
#We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

    #targets = [ClassifierOutputTarget(3)]
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None, 
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)
    
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    #cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    #img = cv2.resize(cam_image ,(700, 330)) 
    #cv2.imwrite(f'{args.method}_cam'f'{args.method}_.jpg',img)
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True,image_weight=0.8)#
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    gb = show_cam_on_image(rgb_img_float, gb, use_rgb=True,image_weight=0.8)#
    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam_1.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')
    #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    #gb = cv2.cvtColor(gb, cv2.COLOR_RGB2BGR)
    #cam_gb = cv2.cvtColor(cam_gb, cv2.COLOR_RGB2BGR)
    dff = DeepFeatureFactorization(model=model, target_layer=target_layers, reshape_transform=reshape_transform,computation_on_concepts=model.head.fc)
    concepts, batch_explanations, concept_scores = dff(input_tensor, n_components=5)
    concept_outputs = torch.softmax(torch.from_numpy(concept_scores), axis=-1).numpy()
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :2]
    
    concept_label_strings = create_labels_legend(concept_outputs, class_dictionary,top_k=1)
    
    print(concept_label_strings)
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.8,
                                                concept_labels=concept_label_strings)
    cam_image = cv2.resize(visualization ,(1400, 660)) 
    gb = cv2.resize(gb ,(700, 330)) 
    cam_gb = cv2.resize(cam_gb ,(700, 330))     
    cv2.imwrite(cam_output_path, cam_image)
    #cv2.imwrite(gb_output_path, gb)
    #cv2.imwrite(cam_gb_output_path, cam_gb)
exit()