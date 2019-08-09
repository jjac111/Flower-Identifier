import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    to_pil = transforms.ToPILImage(mode= 'RGB')
    image = to_pil(image)
    width, height = image.width, image.height
    if height <= width:
        width = int(width*256/height)
        height = 256
    else:
        height = int(height*256/width)
        width = 256
        
    image = image.resize((width, height))
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image = np_image/255
        
    for ij in np.ndindex(np_image.shape[:2]):
        np_image[ij] = (np_image[ij] - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    return np_image.transpose((2,0,1))

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.cpu().numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax


