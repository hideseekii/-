import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import time
import re


#清空快取
torch.cuda.empty_cache()
#rand = np.random.randint(10)


#image in channels
in_channels = 3
#class 分類數
n_classes = 2
    
# hyperparameters
batch_size= 64
n_epochs = 50
lr = 0.001
max_lr = 0.001
weight_decay = 0.0001
grad_clip = 0.1

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')



# define transform function
test_transform = transforms.Compose([

    transforms.Resize((256,256),antialias=True),
    transforms.ToTensor(),
    #transforms.Normalize(mean=0.5, std=0.5)
])

train_transform = transforms.Compose([

    #transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    #transforms.RandomCrop((256,256), padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(degrees=(-45, + 45)),
    transforms.Resize((256,256),antialias=True),
    transforms.ToTensor(),
    #transforms.Normalize(mean=0.5, std=0.5),
    
])


#read
train = ImageFolder("project_b/train80", transform=train_transform, target_transform=None)
val= ImageFolder("project_b/val20", transform=test_transform, target_transform=None)
print(f'class:{train.class_to_idx}')
print(f'train size:{len(train)}')
print(f'val size:{len(val)}')

#dataloader
train_loader = DataLoader(train, batch_size=batch_size,shuffle=True,pin_memory = True)
val_loader = DataLoader(val,pin_memory = True)
print(f'train loader size:{len(train_loader)}')
print(f'val loader size:{len(val_loader)}')


# define LeNet5 model
class LeNet5(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # 建立類神經網路各層
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),  
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        

        
        self.flatten = nn.Flatten()

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=1024*6*6, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=self.n_classes)
        )
        
    def forward(self, x):
        # 定義資料如何通過類神經網路各層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        logits = self.layer6(x)
        return logits
    
    
model = LeNet5(in_channels, n_classes)
model = model.to(device)
#print(model)
for name, params in model.named_parameters():
    print(f'layer name: {name}, parameter shape: {params.shape}')   
# define optimizer & loss function
optimizer = optim.AdamW(model.parameters(), lr=lr
                       ,weight_decay=weight_decay
                       )
sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=n_epochs, 
                                      steps_per_epoch=len(train_loader))
loss_fn = nn.CrossEntropyLoss()


def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    
    model.eval()  # Put model in evaluation mode
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct/total

def train_func(model,train_loader):
    print('\n ------ trainig start ------ \n')
    
    train_loss_list = []
    for epoch in range(n_epochs):
        start = time.time()
        train_loss = 0.
        train_acc = 0.
        for idx, (images, labels) in enumerate(train_loader):
            model.train() 
        
            optimizer.zero_grad(set_to_none=True) # step 1
            
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images) # step 2 (forward pass)
            loss = loss_fn(logits, labels) # step 3 (compute loss)
            
            loss.backward() # step 4 (backpropagation)
            
            if grad_clip: #Gradient clipping
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step() # step 5 (update model parameters)
            #sched.step() #Learning rate scheduling
            
            train_acc = ((logits.argmax(dim = 1) == labels).float().mean())
            train_acc += train_acc/len(train_loader)
            
            train_loss += loss.item()*images.size(0)
                        
        train_loss = train_loss/len(train_loader.sampler)
        train_loss_list.append(train_loss)
        
        val_acc = calculate_accuracy(model,val_loader)
        torch.cuda.synchronize() 
        end = time.time()
        print(f'Epoch: {epoch+1}/{n_epochs} | Loss: {train_loss/(idx+1):.6f} | trainACC: {train_acc:.6f} | valACC: {val_acc:.6f} | time: {end-start:.2f}s ') 
    
    return val_acc 


#訓練集train+val
time_start = time.time()
train_func(model,train_loader)
time_end = time.time()
print(f'Total time:{time_end-time_start:.2f}s')