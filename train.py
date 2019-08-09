
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

# ARGUMENT PARSING
parser = argparse.ArgumentParser(description='Train an artificial neural network to classify images from a given dataset.')
parser.add_argument('data_dir', type=str,
                    help='Directory relative to this script where the training, validation, and testing data are stored.')
parser.add_argument('--save_dir', type=str,
                    help='Optional directory to where the network checkpoint is saved.')
parser.add_argument('--arch', type=str,
                    help='Base model architecture. Examples: densenet121, vgg13, resnet101, etc. Default arch is densenet121.')
parser.add_argument('--learning_rate', type=float,
                    help='Learning rate for the gradient descecent. Default is 0.001')
parser.add_argument('--hidden_units', type=int,
                    help='Number of neurons between the input and output layers.')
parser.add_argument('--epochs', type=int,
                    help='Number of times the model will be trained on the whole training dataset.')
parser.add_argument('--gpu', action='store_true',
                    help='If specified, the network training will take place in the GPU which drastically accelerates the process. If GPU is not available,                       CPU will be used instead.')
args = parser.parse_args()



# VARIABLES
data_dir = args.data_dir
device = 'cpu'
learnrate = args.learning_rate
arch = args.arch
save_dir = args.save_dir
epochs = args.epochs
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('There is no GPU available. Using CPU instead.')
if learnrate:
    if (learnrate < 0) or (learnrate >= 1):
        learnrate = 0.001
        print("Learning rate must be between (0, 1). Assigning 0.001")
else:
    learnrate = 0.001
if not epochs:
    epochs = 30
    

# TRANSFORMS & DATALOADERS
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
transforms_train = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transforms_test_validation = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
datasets_train = datasets.ImageFolder(root= train_dir, transform= transforms_train)
datasets_test = datasets.ImageFolder(root= test_dir, transform= transforms_test_validation)
datasets_validation = datasets.ImageFolder(root= valid_dir, transform= transforms_test_validation)
trainloader = torch.utils.data.DataLoader(datasets_train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets_test, batch_size=32, shuffle=False)
validloader = torch.utils.data.DataLoader(datasets_validation, batch_size=32, shuffle=False)



# LOADING THE MODEL ARCHITECTURE
code = """model = models.{}(pretrained= True)"""
if arch:
    try:
        exec(code.format(arch))
    except:
        exec(code.format('densenet121'))
        arch = 'densenet121'
        print('Specified architecture not valid. Using densenet121 instead.')
else:
    arch = 'densenet121'
    exec(code.format('densenet121'))
uses_fc = ('resnet' in arch) or ('inception' in  arch)
    
    
    
# DEFININING OWN CLASSIFIER AND PARAM FREEZE (classifier has 2 hidden layers always)
in_features = 0
hu = args.hidden_units
out_features = len(datasets_train.classes)
if uses_fc:
    in_features = model.fc.in_features
else:
    in_features = model.classifier.in_features
if hu:
    if hu < 0:
        hu = 623
        print("Negative number of hidden units? Excuse me? Assigning 623.")
else:
    hu = 623
classifier = nn.Sequential(OrderedDict([('0', nn.Linear(in_features, int(hu*2/3))),
                  ('relu0', nn.ReLU()),
                  ('d0', nn.Dropout(p=0.2)),
                  ('1', nn.Linear(int(hu*2/3), int(hu*1/3))),
                  ('relu1', nn.ReLU()),
                  ('d1', nn.Dropout(p=0.2)),
                  ('2', nn.Linear(int(hu*1/3), out_features)),
                  ('logsoftmax', nn.LogSoftmax(dim=1))]))
for param in model.parameters():
    param.requires_grad = False
if uses_fc:
    model.fc = classifier
else:
    model.classifier = classifier
model.to(device)
optimizer = None
if uses_fc:
    optimizer = optim.Adam(model.fc.parameters(), lr= learnrate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr= learnrate)       
criterion = nn.NLLLoss()

                                        


# TRAINING MODEL
train_losses = []
validation_losses = []
accuracies = []
start = time.time()
for e in range(epochs):
    train_loss = 0
    validation_loss = 0
    accuracy = 0

    #save before each training epoch
    save_model(model, in_features, hu, out_features, e, optimizer, arch, datasets_train, save_dir)

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        log_ps = model(inputs)
        loss = criterion(log_ps, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    else:
        with torch.no_grad():
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device) 
                log_ps = model(inputs)
                ps = torch.exp(log_ps)
                validation_loss += criterion(log_ps, labels)
                top_prob, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
                end = time.time()

        accuracy = accuracy*100/len(validloader)

    #Terminate training when accuracy starts going down
    if len(accuracies) != 0: 
        if accuracies[-1] > accuracy:
            print('Accuracy loss detected, stopping training.\n\n\tFINAL CHECKPOINT:') 
            break

    model.train()
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    accuracies.append(accuracy)
    print(f"EPOCH: {e+1}")
    print(f"Training Loss: {train_loss/len(trainloader)}")
    print(f"Validation Loss: {validation_loss/len(validloader)}")
    print(f"Accuracy: {round(float(accuracy), 2)}%")
    print(f"{round(end - start)} seconds\n")      

                                        
              
# TESTING MODEL                                        
test_model(model, testloader, device, criterion=torch.nn.NLLLoss())
                                        
                                        
                                        
                                        
