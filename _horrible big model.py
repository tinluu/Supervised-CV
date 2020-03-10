import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

### Best model so far(97% training, 95% validation)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #conv2d(before, after, kernel_size=, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding = 2)
        torch.nn.init.kaiming_normal_(self.conv1.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        self.conv1_BN = nn.BatchNorm2d(32) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 1)
        torch.nn.init.kaiming_normal_(self.conv2.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        self.conv2_BN = nn.BatchNorm2d(64) 
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding = 1)
        torch.nn.init.kaiming_normal_(self.conv3.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        self.conv3_BN = nn.BatchNorm2d(128) 
        
        self.conv_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(6*6*128, 1024)
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        self.fc1_BN = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 256)
        torch.nn.init.kaiming_normal_(self.fc2.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        self.fc2_BN = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, nclasses)
        torch.nn.init.kaiming_normal_(self.fc3.weight, a=0.01, 
                                      mode='fan_in', nonlinearity="leaky_relu")
        
            
    
    def forward(self, x):
        #Convolution Layer 1
        #_max_pool2d(input, kernel_size, stride=None, padding=0)
        x = self.conv1(x)
        x = self.conv1_BN(x)
        x = F.leaky_relu(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, kernel_size = 3, stride = 1)
        
        # Convolution Layer 2
        # dropout(input, p=0.5, training=True, inplace=False)
        x = self.conv2(x)
        x = self.conv2_BN(x)
        x = F.leaky_relu(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, kernel_size = 3, stride = 1)
        
        # Convolution Layer 3
        x = self.conv3(x)
        x = self.conv3_BN(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size = 3, stride = 1)
        
        # size 6x6
        x = F.adaptive_avg_pool2d(x, output_size = 6)
        #x = F.adaptive_max_pool2d(x, output_size = 6)
        
        #Flattening to Fully Connected Layer
        x = torch.flatten(x,1)
        
        #Fully Connected Layer 1
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = self.fc1_BN(x)
        x = F.leaky_relu(x)
        
        #Fully Connected Layer 2
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc2_BN(x)
        x = F.leaky_relu(x)
        
        #Fully Connected Layer 
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim = 1)
