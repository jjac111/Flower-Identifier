import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import PIL
import numpy as np
from collections import OrderedDict
import time
import argparse
from train_utils import *
from image_utils import *

def predict(image_path, model, topk=5, cat_to_name=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = PIL.Image.open(image_path)
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    image = torch.tensor(process_image(image))
    
    image = image.unsqueeze(dim=0)
    image = image.to(device).float()
    model = model.to(device).float()
    
    
    log_ps = model(image)
    ps = torch.exp(log_ps)
    
    top_p, top_classes = ps.topk(topk, dim=1)
    
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}
    
    labels = []
    if cat_to_name:
        for clss in top_classes[0]:
            labels.append(cat_to_name[idx_to_class[clss.item()]])
    else:
        for clss in top_classes[0]:
            labels.append(str(idx_to_class[clss.item()]))
    return top_p, labels

parser = argparse.ArgumentParser(description='Use a pretrained Artificial Neural Network to predict the type of a flower image input.')
parser.add_argument('input', type=str,
                    help='Directory where of image input (Only tested for JPG).')
parser.add_argument('--checkpoint', type=str,
                    help='Directory of the ANN training checkpoint. If not specified, \'checkpoint.pth\' will be searched in the local directory.')
parser.add_argument('--top_k', type=int,
                    help='The top number of category probabilities calculated for the input.')
parser.add_argument('--category_names', type=str,
                    help='Learning rate for the gradient descecent. Default is 0.001')
parser.add_argument('--gpu', action='store_true',
                    help='If specified, the network training will take place in the GPU which drastically accelerates the process. If GPU is not available,                       CPU will be used instead.')
args = parser.parse_args()


image_path = args.input
cat_to_name = None
device = 'cpu'
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
if args.category_names:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    print('No category file specified. Categorical labels will not be translated.\n')
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('There is no GPU available. Using CPU instead.\n')
if not checkpoint:
    checkpoint = 'checkpoint.pth'
if not top_k:
    top_k = 5    

model = load_model(checkpoint)
model.eval()        
   
    
probs, class_names = predict(image_path, model, top_k, cat_to_name)

print(class_names)
print('\n')
print(probs)

#PLOTTING
'''
x = np.arange(len(class_names))
y = probs.tolist()[0]

plt.subplot(2,1,1)
ax = plt.gca()
imshow(image= image, ax= ax)
plt.show()
plt.subplot(2,1,2)
plt.barh(x, y, align='center')
plt.yticks(x, class_names)
plt.xticks(rotation=30)
ax = plt.gca()
ax.set_xlim(min(y)*0.95, max(y)*1.05)
plt.show()
'''