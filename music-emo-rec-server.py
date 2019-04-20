import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import csv
import time
import _thread
import datetime
from torch.utils.data import Dataset, DataLoader


start_datetime = datetime.datetime.now()
start_time = time.time()
start_clock = time.clock()
with open('record', 'a') as f:
    f.write('start datetime: {}\nstart time: {}\n\
        start clock: {}\n'.format(start_datetime, start_time, start_clock))


# train_rate = 0.8
train_slice_num = 2223  # 用来训练的曲子数
pic_len = 256           # 图片长度
batch_size = 50         
epoch_num = 10
# input_num = 2
interval = 500            # 窗口间隔
part_data_num = 1000     # 每一次训读入的数据
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

# print('start')
with open('record', 'a') as f:
    f.write('start\n')


# print('slice_num is {}'.format(len(input_slice)))
# print('music input end, start to label')
# with open('record', 'a') as f:
#     f.write('slice_num is {}\nmusic input end, start to label\n'.format(slice_num))

# 读入label
label_file = open('prodLabels.csv', 'r')
label_reader = csv.reader(label_file)
#print(list(label_reader)[0:5])
reader_list = list(label_reader)
create_labels = True
for line in reader_list:
    if line == []:
        continue
    if create_labels:
        labels = [list(map(int, line))]
        create_labels = False
    else:
        labels.append(list(map(int, line)))
label_len = len(labels[0])
#labels = list(map(int, label_reader))[0:20]
# print(labels[input_num-1])


# 对每一行取出多个128*128数据执行傅里叶变换，把时频数据分别放入不同的channel，就得到了一张图片，多次循环得到多张图片
# train_num = math.ceil(slice_num*train_rate)
# create_train_tensor = True
# create_test_tensor = True
train_data = torch.Tensor(part_data_num, 2, pic_len, pic_len)
test_data = torch.Tensor(part_data_num, 2, pic_len, pic_len)
train_label = torch.Tensor(part_data_num, label_len)
test_label = torch.Tensor(part_data_num, label_len)
train_or_test = 'train'
true_data_len = 0
train_data_is_left = True
test_data_is_left = True
train_data_has_refreshed = False
test_data_has_refreshed = False

def transfer_data():
    # global slice_num
    # global input_slice
    global pic_len
    global train_or_test
    global train_data
    global test_data
    global interval
    global true_data_len
    global part_data_num
    global train_slice_num
    global train_data_is_left
    global test_data_is_left
    global train_data_has_refreshed
    global test_data_has_refreshed

    # 读入音频数据，计算数据行数
    input_file = open('prodAudios_v2.txt', 'r')
    input_lines = input_file.readlines()
    # create_inp_list = True
    slice_num = 0
    for line in input_lines:
        slice_num += 1
        line = line.split(' ')
        line.pop()
        line = list(map(int, line))
        line_len = len(line)
        with open('record', 'a') as f:
            f.write(str(line_len) + '\n')
        # print(line_len)
        sample_start = 0
        sample_len = pic_len*pic_len
        while sample_start + sample_len <= line_len:
            time_data = line[sample_start: sample_start + sample_len]
            freq_data = abs(np.fft.fft(time_data)/sample_len)
            time_data = np.array(time_data).reshape(pic_len, pic_len)
            freq_data = np.array(freq_data).reshape(pic_len, pic_len)
            sample_start += interval
            if train_or_test == 'train':
                train_data[true_data_len] = torch.Tensor([time_data, freq_data])
                train_label[true_data_len] = torch.Tensor(labels[slice_num - 1])
            else:
                test_data[true_data_len] = torch.Tensor([time_data, freq_data])
                test_label[true_data_len] = torch.Tensor(labels[slice_num - 1])
            true_data_len += 1
            if true_data_len == part_data_num:
                if train_or_test == 'train':
                    train_data_has_refreshed = True
                else:
                    test_data_has_refreshed = True
            while true_data_len == part_data_num:
                time.sleep(1)
        if slice_num >= train_slice_num - 1:
            train_or_test = 'test'
            train_data_is_left = False
    input_file.close()    
    test_data_is_left = False

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
        linear_len = int(((pic_len - 4) / 2 - 4) / 2 - 6 - 4)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(8 * linear_len * linear_len, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = x.view(-1, 8 * self.linear_len * self.linear_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


net = Net()
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
net.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

def train():
    global epoch_num
    global optimizer
    global net
    global criterion
    global train_data
    global train_label
    global batch_size
    global train_data_has_refreshed
    global true_data_len
    global part_data_num
    global device
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        while train_data_is_left:
            if not train_data_has_refreshed:
                time.sleep(0.5)
                continue
            if true_data_len < part_data_num:
                batch_size = 1
            train_data_has_refreshed = False
            true_data_len = 0
            tmp_train_data = train_data.clone()
            tmp_train_label = train_label.clone()
            train_loader = DataLoader(dataset=TimeFreqDataset(tmp_train_data, tmp_train_label),
                         batch_size=batch_size, shuffle=True)
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                # outputs = torch.round(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #       (epoch + 1, i + 1, running_loss / 500))
                    with open('record', 'a') as f:
                        f.write('[%d, %5d] loss: %.3f\n\n' %
                            (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0


    # print('Finished Training')
    with open('record', 'a') as f:
        f.write('Finished Training\n')


def test():
    global test_data
    global test_label
    global batch_size
    global test_data_is_left
    global net
    global test_data_has_refreshed
    global true_data_len
    global part_data_num
    global device
    correct = 0
    correct_v2 = 0
    total = 0
    loss = 0
    with torch.no_grad():
        while test_data_is_left:
            if not test_data_has_refreshed:
                time.sleep(0.5)
                continue
            if true_data_len < part_data_num:
                batch_size = 1
            test_data_has_refreshed = False
            true_data_len = 0
            tmp_test_data = test_data.clone()
            tmp_test_label = test_label.clone()
            test_loader = DataLoader(dataset=TimeFreqDataset(tmp_test_data, tmp_test_label),
                         batch_size=batch_size, shuffle=True)
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                outputs = torch.round(outputs)
                #labels = labels.float()
                total += labels.size(0)
                ########################################
                outputs[outputs < 0] = 0
                outputs[outputs > 1] = 1
                ########################################
                correct += (outputs.data == labels).sum().item()
                # loss += abs(outputs.data - labels).sum().item()
                tmp_loss = abs(outputs.data - labels).sum().item()
                if tmp_loss == 0:
                    correct_v2 += 1
                loss += tmp_loss

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print('Loss of the network: {}'.format(loss))
    print('My_Accuracy of the network on the test images: %d %%' % (
        100 * correct_v2 / total))
    with open('record', 'a') as f:
        f.write('Accuracy of the network on the test images: %d %%\n' % (
            100 * correct / total))
    with open('record', 'a') as f:
        f.write('Loss of the network: {}\n'.format(loss))
    with open('record', 'a') as f:
        f.write('My_Accuracy of the network on the test images: %d %%\n' % (
            100 * correct_v2 / total))



if __name__ == "__main__":
    _thread.start_new_thread(transfer_data, ())
    train()
    test()
    end_datetime = datetime.datetime.now()
    end_time = time.time()
    end_clock = time.clock()
    with open('record', 'a') as f:
        f.write('end datetime: {}\nend time: {}\nend clock: {}\n\
            dur_datetime: {}\ndur_time: {}\ndur_clock: {}\n\n\n\n'.
            format(end_datetime, end_time, end_clock, 
            (end_datetime - start_datetime).seconds,
            end_time - start_time, end_clock - start_clock))


