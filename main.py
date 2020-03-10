from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=277, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from model import Net
model = Net()
model = model.to(device)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
#                 eps=1e-08, weight_decay=0.01, amsgrad=False )

optimizer = optim.AdamW(model.parameters(), 
                        lr=0.001, betas=(0.9, 0.999), 
                        eps=1e-08, weight_decay=0.1, amsgrad=False)

#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.9, last_epoch=-1)

torch.backends.cudnn.benchmark = True

train_loss_record = []
train_acc_record = []

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
            train_loss_record.append(loss)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_acc_record.append(100. * correct / len(val_loader.dataset))
        
        
        
    loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
   


val_acc_record = []

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    val_acc_record.append(correct/len(val_loader.dataset))
        

    validation_loss /= len(val_loader.dataset)
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    scheduler.step()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')

import matplotlib.pyplot as plt
plt.plot(val_acc_record)
plt.ylabel('loss')
plt.show()



