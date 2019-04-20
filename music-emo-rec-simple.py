import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import csv
import datetime
import time
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image

start_datetime = datetime.datetime.now()
start_time = time.time()
start_clock = time.clock()
with open('record', 'a') as f:
    f.write('start datetime: {}\nstart time: {}\n\
        start clock: {}\n'.format(start_datetime, start_time, start_clock))
    

train_rate = 0.8
pic_len = 128
batch_size = 50
epoch_num = 10
input_num = 3000

# print('start')
with open('record', 'a') as f:
    f.write('start\n')

#读入音频数据，计算数据行数
input_file = open('prodAudios_server', 'r')
input_lines = input_file.readlines()
create_inp_list = True
slice_num = 0
for line in input_lines:
	slice_num += 1
	if slice_num > input_num:
		slice_num = input_num
		break
	line = line.split(' ')
	line.pop()
	line = list(map(int, line))
	if create_inp_list:
		input_slice = [line]
		create_inp_list = False
	else:
		input_slice.append(line)

# print('slice_num is {}'.format(slice_num))
# print('music input end, start to label')
with open('record', 'a') as f:
    f.write('slice_num is {}\nmusic input end, start to label\n'.format(slice_num))

# 读入label
label_file = open('prodLabels_server.csv', 'r')
label_reader = csv.reader(label_file)
#print(list(label_reader)[0:5])
reader_list = list(label_reader)
labels = [[int(reader_list[i][j]) for j in range(len(reader_list[i]))] for i in range(input_num)]
#labels = list(map(int, label_reader))[0:20]
# print(labels[input_num-1])


# 对每一行取出多个128*128数据执行傅里叶变换，把时频数据分别放入不同的channel，就得到了一张图片，多次循环得到多张图片
train_num = math.ceil(slice_num*train_rate)
create_train_tensor = True
create_test_tensor = True
for i in range(slice_num):
    # print(i)
    line = input_slice[i]
    line_len = len(line)
    sample_start = 0
    sample_len = pic_len*pic_len
    while sample_start + sample_len <= line_len - 1:
        time_data = line[sample_start: sample_start + sample_len]
        freq_data = abs(np.fft.fft(time_data)/sample_len)
        time_data = np.array(time_data).reshape(pic_len, pic_len)
        freq_data = np.array(freq_data).reshape(pic_len, pic_len)
        sample_start += 128
        if i < train_num:
            if create_train_tensor:
                train_data = torch.Tensor([[time_data, freq_data]])
                train_label = torch.Tensor([labels[i]])
                create_train_tensor = False
            else:
                train_data = torch.cat((train_data, torch.Tensor([[time_data, freq_data]])))
                train_label = torch.cat((train_label, torch.Tensor([labels[i]])))
        else:
            if create_test_tensor:
                test_data = torch.Tensor([[time_data, freq_data]])
                test_label = torch.Tensor([labels[i]])
                create_test_tensor = False
            else:
                test_data = torch.cat((test_data, torch.Tensor([[time_data, freq_data]])))
                test_label = torch.cat((test_label, torch.Tensor([labels[i]])))

# 重写Dataset类
class TimeFreqDataset(Dataset):

    def __init__(self, data, label):
        self.len = len(data)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len
# 创建data_loader
train_loader = DataLoader(dataset=TimeFreqDataset(train_data, train_label),
                         batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=TimeFreqDataset(test_data, test_label),
                         batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8 * 48 * 48, 120)
        self.fc2 = nn.Linear(120, 18)
        # self.fc3 = nn.Linear(336, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = x.view(-1, 8 * 48 * 48)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


net = Net()

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

for epoch in range(epoch_num):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #outputs = round(outputs)
        outputs = torch.round(outputs)
        #outputs = outputs.long()
        #labels = labels.float()
        #labels = labels.long()
        #print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / 2000))
            with open('record', 'a') as f:
                f.write('[%d, %5d] loss: %.3f\n\n' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


# print('Finished Training')


correct = 0
total = 0
loss = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        outputs = torch.round(outputs)
        #labels = labels.float()
        total += labels.size(0)
        #correct += (outputs.data == labels).sum().item()
        loss += abs(outputs.data - labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
    # 100 * loss / total))
with open('record', 'a') as f:
    f.write('Accuracy of the network on the test images: %d %%' % (
    100 * loss / total))

end_datetime = datetime.datetime.now()
end_time = time.time()
end_clock = time.clock()
with open('record', 'a') as f:
    f.write('end datetime: {}\nend time: {}\nend clock: {}\n\
        dur_datetime: {}\ndur_time: {}\ndur_clock: {}\n\n\n\n'.
        format(end_datetime, end_time, end_clock, 
        (end_datetime - start_datetime).seconds,
        end_time - start_time, end_clock - start_clock))

