# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

# Residual block
# nn.Module: 딥러닝과 관련된 모든 내용을 담고 있는 기본 class (forward, backward 사용 가)
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block have different output size
    #we use class attribute expansion to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            # 1x1 convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 3x3 convolution
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 1x1 convolution
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# ResNet
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        """_summary_

        Args:
            block: block type, basic block or bottle neck block
            num_block: how many blocks per layer 
            num_classes: The number of classification class Defaults to 100.
        """        
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        #we use a different input size than the original paper
        #so conv2_x's stride is 1 (pooling 대신)
        #stride1: 크기 유지, 세밀 특성 추출, stride2: 다운샘플링, 계층 전환시 사용
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        # 마지막에 global average로 사용
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

 
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3]).to(device)

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3]).to(device)

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3]).to(device)

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3]).to(device)

model = ResNet18()
#model = ResNet34()
#model = ResNet50()
#model = ResNet101()
#model = ResNet152()

criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, verbose=True)

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# train epoch
for epoch in range(num_epochs):
  # Use Dropout layer and Batch Norm layer
  model.train()
  correct, count = 0, 0
  train_loss = 0
  for batch_idx, (images, labels) in enumerate(train_loader, start=1):
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      optimizer.zero_grad()
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      _, preds = torch.max(output, 1)
      count += labels.size(0)
      correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)

      
      # Train accuracy와 loss 저장
      train_losses.append(train_loss / count)
      train_accuracies.append(correct / count)
      
      print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_loader)}\tTrain accuracy: {round((correct/count)*100, 4)} \tTrain Loss: {round((train_loss/count)*100, 4)}")


  # Don't use Dropout layer and Batch Norm layer
  model.eval()
  correct, count = 0, 0
  valid_loss = 0
  # 내부 코드 역전파 수행하지 않음. -> 모델 영향 X, 계산 빨라짐.
  with torch.no_grad():
      for batch_idx, (images, labels) in enumerate(test_loader, start=1):
          images, labels = images.to(device), labels.to(device)
          output = model(images)
          loss = criterion(output, labels)
          valid_loss += loss.item()
          _, preds = torch.max(output, 1)
          count += labels.size(0)
          correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)
          
          # Validation accuracy와 loss 저장
          valid_losses.append(valid_loss / count)
          valid_accuracies.append(correct / count)

          print (f"[*] Step: {batch_idx}/{len(test_loader)}\tValid accuracy: {round((correct/count), 4)} \tValid Loss: {round((valid_loss/count), 4)}")

# Train Loss와 Accuracy 그래프
plt.plot(train_losses, label='Train Loss')
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()

# Validation Loss와 Accuracy 그래프
plt.plot(valid_losses, label='Validation Loss')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Validation Loss and Accuracy')
plt.legend()
plt.show()
