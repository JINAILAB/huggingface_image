import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
from functools import partial
import torch


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          method: Callable=GradCAMPlusPlus):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
        return np.hstack(results)
    
    
def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")
    

def get_target_layer(model): 
    """
    모델의 종류에 따라 target_layer를 추출합니다. dff_layer_dff와 target_layer_gradcam을 return
    """
    # Swin Transformer
    if hasattr(model, 'swin'):
        return model.swin.layernorm, model.swin.layernorm
    # ConvNeXt
    elif hasattr(model, 'convnext'):
        return model.convnext.encoder.stages[-1].layers[-1], model.convnext.encoder.stages[-1].layers[-1]
    # resnet
    elif hasattr(model, 'resnet'):
        return model.resnet.encoder.stages[-1].layers[-1], model.resnet.encoder.stages[-1].layers[-1]
    # vit
    elif hasattr(model, 'vit'):
        return model.vit.layernorm, model.vit.encoder.layer[-2].output
    # regnet
    elif hasattr(model, 'regnet'):
        return model.regnet.encoder.stages[-1], model.regnet.encoder.stages[-1]
    # cvt
    elif hasattr(model, 'cvt'):
        return model.cvt.encoder.stages[-1].layers[-1], model.cvt.encoder.stages[-1].layers[-2]
    # 기타 모델에 대해서는 지원 X
    else:
        return None, None
    
    
    
def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_convnext_huggingface(tensor, model):
    tensor = tensor.transpose(1, 2).transpose(2, 3)
    norm = model.convnext.layernorm(tensor)
    return norm.transpose(2, 3).transpose(1, 2)

def reshape_gradcam_transform_convnext_huggingface(tensor, model):
    tensor = tensor.transpose(1, 2).transpose(2, 3)
    return tensor.transpose(2, 3).transpose(1, 2)

def reshape_transform_cvt_huggingface(tensor, model, width, height):
    tensor = tensor[:, 1 :, :]
    tensor = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(-1))
    
    norm = model.layernorm(tensor)
    return norm.transpose(2, 3).transpose(1, 2)

def reshape_gradcam_transform_cvt_huggingface(tensor, model, width, height):
    tensor = tensor[:, 1 :, :]
    tensor = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(-1))
    return tensor.transpose(2, 3).transpose(1, 2)

def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0],
                                   12, 12, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

    
    
def reshape_transform_generic(tensor, model, img_tensor):
    # Determining the dimensions
    width, height = img_tensor.shape[2], img_tensor.shape[1]
    batch, features, tensor_height, tensor_width = tensor.shape

    # Swin Transformer
    if hasattr(model, 'swin'):
        reshape_transform = partial(swinT_reshape_transform_huggingface(tensor, tensor_width // 32, tensor_height // 32))
        
        return reshape_transform, reshape_transform

    # ConvNeXt
    elif hasattr(model, 'convnext'):
        reshape_transform = partial(reshape_transform_convnext_huggingface, model=model)
        gradcam_transform = partial(reshape_gradcam_transform_convnext_huggingface, model=model)
        
        return reshape_transform, gradcam_transform

    # CVT
    elif hasattr(model, 'cvt'):
        reshape_transform = partial(reshape_transform_cvt_huggingface,
                            model=model,
                            width=img_tensor.shape[2]//16,
                            height=img_tensor.shape[1]//16)
        gradcam_transform = partial(reshape_gradcam_transform_cvt_huggingface,
                            model=model,
                            width=img_tensor.shape[2]//16,
                            height=img_tensor.shape[1]//16)
        return reshape_transform, gradcam_transform
    
    # vit
    elif hasattr(model, 'vit'):
        reshape_transform = reshape_transform_vit_huggingface
        return reshape_transform, reshape_transform

    # resnet, regnet
    else:
        # Handle unknown model or return a default transformation
        return None, None