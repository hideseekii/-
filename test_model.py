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
lr = 0.00001
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


class Brian(nn.Module):
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
    
    
model = Brian(in_channels, n_classes)
model = model.to(device)
#print(model)
for name, params in model.named_parameters():
    print(f'layer name: {name}, parameter shape: {params.shape}')   
# define optimizer & loss function
optimizer = optim.AdamW(model.parameters(), lr=lr
                       ,weight_decay=weight_decay
                       )
# sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=n_epochs, 
#                                       steps_per_epoch=len(train_loader))
loss_fn = nn.CrossEntropyLoss()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
early_stopper = EarlyStopper(patience=7)



def train(model, train_loader, device, optimizer, loss_fn):
    model.train()
    train_loss = 0.
    n_corrects = 0
    total = 0
    for idx,(images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = outputs.max(1)
        n_corrects += predictions.eq(labels).sum().item()
        total += labels.size(0)
        loss = loss_fn(outputs, labels)

        loss.backward()
        if grad_clip: #Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        #sched.step()

        train_loss += loss.item()*images.size(0)
        current_lr = optimizer.param_groups[0]['lr']

    train_loss = train_loss/len(train_loader.sampler)

    return train_loss, n_corrects/total,current_lr



@torch.no_grad()
def validate(model, valid_loader, device, loss_fn):
    model.eval()
    n_corrects = 0
    total = 0
    valid_loss = 0.
    for idx, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        _, predictions = outputs.max(1)
        n_corrects += predictions.eq(labels).sum().item()
        total += labels.size(0)
        valid_loss += loss.item()*images.size(0)

    valid_accuracy = n_corrects / total

    valid_loss = valid_loss/len(valid_loader.sampler)

    return valid_loss, valid_accuracy



from torch.optim import lr_scheduler
import time
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
is_valid_available = True
#  #scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9 ** epoch)
#  #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)



train_loss_list = []
valid_loss_list = []

train_accuracy_list = []
valid_accuracy_list = []

early_stopper = EarlyStopper(patience=7)

for epoch in range(n_epochs):
    start_time = time.time()
    
    training_loss, training_accuracy, c_lr = train(model, train_loader,  device,optimizer, loss_fn)
    valid_loss, valid_accuracy = validate(model, val_loader, device, loss_fn)

    train_loss_list.append(training_loss)
    valid_loss_list.append(valid_loss)

    train_accuracy_list.append(training_accuracy)
    valid_accuracy_list.append(valid_accuracy)
  
    if scheduler is not None and is_valid_available:
        scheduler.step(valid_loss)
    elif scheduler is not None:
        scheduler.step()
    # sched.step() 
        
    end_time = time.time()
    epoch_time = end_time - start_time
    
    print(f"[Epoch {epoch+1}/{n_epochs}] trainacc:{training_accuracy:.3f} | trainloss:{training_loss:.4f} | valacc:{valid_accuracy:.3f} | valloss:{valid_loss:.4f} | lr:{c_lr:.8f} | t:{epoch_time:.0f}s")

    if early_stopper.early_stop(valid_loss):
        break

from sklearn.metrics import confusion_matrix, classification_report, f1_score
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(all_labels, all_preds)
print("Classification Report:")
print(class_report)

f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"F1 Score: {f1}")

torch.save(model, 'model/test_model.pt')