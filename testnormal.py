import torch
import torch.nn as nn
# import torch.gather as gather
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)

    def forward(self, x):
        print(x)
        print(x[0])
        b = x[0][0][0][0]
        print(b)
        return x

a = torch.Tensor([[[[1,1],[2,2]],[[3,3],[4,4]]],[[[5,5],[6,6]],[[7,7],[8,8]]]])
# print(a)
net = Net()
net(a)

