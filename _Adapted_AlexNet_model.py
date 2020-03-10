import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(500, 50)
        #self.fc2 = nn.Linear(50, nclasses)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 1),
            
            nn.Conv2d(64, 192, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 1),
            
            nn.Conv2d(192, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size = 3, stride = 1),
            
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size = 3, stride = 1),
            
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        #self.avgpool = nn.AdaptiveAvgPool2d((6,6)) #decide on the size of features
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*14*14, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096,nclasses),
        )
        

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 500)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return F.log_softmax(x)
