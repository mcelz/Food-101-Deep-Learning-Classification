import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class baseline(nn.Module):
     def __init__(self):
        super(custom, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.BN1=nn.BatchNorm2d(64)
        self.drop_out3=nn.Dropout(p=0.2)
        
        self.conv2= nn.Conv2d(64, 128, kernel_size=3)
        self.BN2=nn.BatchNorm2d(128)
        
        
        self.conv3= nn.Conv2d(128, 256, kernel_size=3)
        self.BN3=nn.BatchNorm2d(256)
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((107, 107))
        
        self.conv4= nn.Conv2d(256, 256, kernel_size=3)
        self.BN4=nn.BatchNorm2d(256)
        
        
        self.conv5= nn.Conv2d(256, 512, kernel_size=3)
        self.BN5=nn.BatchNorm2d(512)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv6= nn.Conv2d(512, 128, kernel_size=3,stride=2)
        self.BN6=nn.BatchNorm2d(128)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        
        # Compute the input units using the formula given in discussion session 128*63*63
        self.fc1=nn.Linear(128, 128)
        self.drop_out=nn.Dropout()
        self.fc2=nn.Linear(128,20)
        
    def forward(self, x):
        # layer1
        x=self.conv1(x)
        x=self.BN1(x)
        x=F.relu(self.drop_out3(x))
        
        # Layer2
        x=self.conv2(x)
        x=self.BN2(x)
        x=F.relu(x)
        
        # Layer3
        x=self.conv3(x)
        x=self.BN3(x)
        x=F.relu(x)
        
        x = self.avg_pool1(x)
        
        # Layer4
        x=self.conv4(x)
        x=self.BN4(x)
        x=F.relu(x)
        x=F.max_pool2d(x,3)
        
        # Layer5
        x=self.conv5(x)
        x=self.BN5(x)
        x=F.relu(x)
        x=self.max3(x)
        
        # Layer6
        x=self.conv6(x)
        x=self.BN6(x)
        x=F.relu(x)
        x=self.max4(x)
        
        x=self.avg_pool(x)
        # fully connected layer 1
        x=x.view(-1,128) # faltten the input
        x=self.fc1(x)
        x=F.relu(self.drop_out(x))
        # fully connected layer 2
        x=self.fc2(x)
        return x

class custom(nn.Module):
    def __init__(self):
        super(custom, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.BN1=nn.BatchNorm2d(64)
        self.drop_out3=nn.Dropout(p=0.2)
        
        self.conv2= nn.Conv2d(64, 128, kernel_size=3)
        self.BN2=nn.BatchNorm2d(128)
        
        
        self.conv3= nn.Conv2d(128, 256, kernel_size=3)
        self.BN3=nn.BatchNorm2d(256)
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((107, 107))
        
        self.conv4= nn.Conv2d(256, 256, kernel_size=3)
        self.BN4=nn.BatchNorm2d(256)
        
        
        self.conv5= nn.Conv2d(256, 512, kernel_size=3)
        self.BN5=nn.BatchNorm2d(512)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv6= nn.Conv2d(512, 128, kernel_size=3,stride=2)
        self.BN6=nn.BatchNorm2d(128)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        
        # Compute the input units using the formula given in discussion session 128*63*63
        self.fc1=nn.Linear(128, 128)
        self.drop_out=nn.Dropout()
        self.fc2=nn.Linear(128,20)
        
    def forward(self, x):
        # layer1
        x=self.conv1(x)
        x=self.BN1(x)
        x=F.relu(self.drop_out3(x))
        
        # Layer2
        x=self.conv2(x)
        x=self.BN2(x)
        x=F.relu(x)
        
        # Layer3
        x=self.conv3(x)
        x=self.BN3(x)
        x=F.relu(x)
        
        x = self.avg_pool1(x)
        
        # Layer4
        x=self.conv4(x)
        x=self.BN4(x)
        x=F.relu(x)
        x=F.max_pool2d(x,3)
        
        # Layer5
        x=self.conv5(x)
        x=self.BN5(x)
        x=F.relu(x)
        x=self.max3(x)
        
        # Layer6
        x=self.conv6(x)
        x=self.BN6(x)
        x=F.relu(x)
        x=self.max4(x)
        
        x=self.avg_pool(x)
        # fully connected layer 1
        x=x.view(-1,128) # faltten the input
        x=self.fc1(x)
        x=F.relu(self.drop_out(x))
        # fully connected layer 2
        x=self.fc2(x)
        return x

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        temp=models.resnet18(pretrained=True)
        num_ftrs = temp.fc.in_features
        temp.fc = nn.Linear(num_ftrs, 20)
        model_ft = temp.to("cuda:0")
        self.model=model_ft
        
    def forward(self, x):
        x=self.model(x)
        return x

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()

        temp=models.vgg16_bn(pretrained=True)

        # change the last layer
        temp.classifier[6] = nn.Linear(in_features=4096, out_features=20, bias=True)
        
        # unfreeze the final layer this time for extra tuning
        temp.classifier[6].weight.requires_grad = True
        temp.classifier[6].bias.requires_grad = True
        
        model_ft = temp.to("cuda:0")
        self.model=model_ft
        
    def forward(self, x):
        x=self.model(x)
        return x

def get_model(args):
    model = None
    return model
