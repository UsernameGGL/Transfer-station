import csv
import numpy as np
import os

pic_len = 256

output_dir = '../data/cal500/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# 读入音频数据，计算数据行数
input_file = open('prodAudios_v2.txt', 'r')
input_slice = input_file.readlines()
create_inp_list = True
slice_num = 0
length = len(input_slice)
for i in range(length):
	input_slice[i] = input_slice[i].split(' ')
	if input_slice[i][len(input_slice[i]) - 1] == '\n':
		input_slice[i].pop()
		print(i)
	input_slice[i] = list(map(int, input_slice[i]))

# 对每一行取出多个128*128+1数据执行傅里叶变换，把时频数据分别放入不同的channel，就得到了一张图片，多次循环得到多张图片
create_train_tensor = True
create_test_tensor = True
for i in range(length):
	print('正在处理第' + str(i+1) + '个文件')
	# print(i)
	line = input_slice[i]
	line_len = len(line)
	print(line_len)
	sample_start = 0
	sample_len = pic_len*pic_len + 1
	tmp_i = i + 1
	file_name = '0'
	while tmp_i < 10000:
		tmp_i *= 10
		file_name += '0'
	file_name += str(i+1) + '.csv'
	out_file = open(output_dir + file_name, 'w')
	writer = csv.writer(out_file)
	while sample_start + sample_len <= line_len:
		time_data = line[sample_start: sample_start + sample_len]
		freq_data = abs(np.fft.fft(time_data)/sample_len)
		sample_start += sample_len
		row_data = np.append(time_data, freq_data)
		writer.writerow(row_data)
	writer.writerow(line[sample_start: line_len])
	out_file.close()
		
