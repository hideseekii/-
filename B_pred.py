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
n_epochs = 1
lr = 0.001

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



#read
test= ImageFolder("../../TOPIC/ProjectB/B_testing", transform=test_transform, target_transform=None)
print(f'val size:{len(test)}')

#dataloader
test_loader = DataLoader(test,pin_memory = True)
print(f'val loader size:{len(test_loader)}')


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
    
    
model = torch.load('model/B_model.pt') 
model = model.to(device)




from pathlib import Path
df1=pd.DataFrame(test_loader.dataset.imgs)
df1.columns=['image_name','label']
for index, datarow in df1.iterrows():
    df1.loc[index,'image_name'] = Path(df1.loc[index,'image_name']).name

df1

print(df1)


model.eval()
predicted_labels = []

img_count = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        data = int(predictions)
        df1.loc[img_count,'label'] = data

        img_count = img_count+1
        
df1 = df1.sort_values(['image_name'],ascending=True)
df1 = df1.set_index('image_name')
df1.to_csv('submit/TOPIC/ProjectB/112060_projectB_ans.csv')