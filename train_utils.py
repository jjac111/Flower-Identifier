
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict

def save_model(model, in_features, hidden_neurons, out_features, epochs, optimizer, arch, dataset, save_dir=None):
    model.class_to_idx = dataset.class_to_idx

    model_checkpoint = {'state_dict': model.state_dict(),
                        'input': in_features,
                        'hidden': [int(hidden_neurons*2/3), int(hidden_neurons*1/3)],
                        'output': out_features,
                        'epochs': epochs,
                        'optimizer_state': optimizer.state_dict(),
                        'class_to_idx': model.class_to_idx,
                        'arch': arch}
    if not save_dir:
        save_dir = ''
    torch.save(model_checkpoint, save_dir+'checkpoint.pth')
    
def test_model(model, testloader, device='cpu', criterion=torch.nn.NLLLoss()):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device) 
            log_ps = model(inputs)
            ps = torch.exp(log_ps)
            test_loss += criterion(log_ps, labels)
            top_prob, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))
    print(f"Test Loss: {test_loss/len(testloader)}")
    print(f"Tested Accuracy: {round(float(accuracy*100/len(testloader)), 2)}%")

def load_model(filename, path= ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cp = torch.load(path+filename, map_location=lambda storage, loc: storage)
    layers = OrderedDict()
    
    layers['0'] = nn.Linear(model_cp['input'], model_cp['hidden'][0])
    layers['relu0'] = nn.ReLU()
    layers['d0'] = nn.Dropout(p=0.2)
    for i in range(len(model_cp['hidden'])):
        if i != len(model_cp['hidden']) -1:
            layers[str(i+1)] = nn.Linear(model_cp['hidden'][i], model_cp['hidden'][i+1])
            layers['relu'+str(i+1)] = nn.ReLU()
            layers['d'+str(i+1)] = nn.Dropout(p=0.2)
    else:
        layers[str(len(model_cp['hidden']))] = nn.Linear(model_cp['hidden'][-1], model_cp['output'])
        layers['logsoftmax'] = nn.LogSoftmax(dim=1)
    
    #Works only for this type of architecture for now
    model = models.densenet121(pretrained= True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(layers)
    model.load_state_dict(model_cp['state_dict'], True)
    model.class_to_idx = model_cp['class_to_idx']
    
    return model.to(device)