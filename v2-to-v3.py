import csv
import numpy as np
import os
import statistics

pic_len = 256

output_dir = 'C:/Users/guoliang.gao1/OneDrive - University of Calgary/data/cal500/music-data/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# 读入音频数据，计算数据行数
input_file = open('../data/cal500/prodAudios_v2.txt', 'r')
input_slice = input_file.readlines()
create_inp_list = True
slice_num = 0
length = len(input_slice)
for i in range(length):
	line = input_slice[i].split(' ')
	if line[len(line) - 1] == '\n':
		line.pop()
	line = list(map(int, line))
	print('正在处理第' + str(i + 1) + '个文件')
	# print(i)
	# line = input_slice[i]
	line_len = len(line)
	print(line_len)
	sample_start = 0
	sample_len = pic_len * pic_len
	tmp_i = i + 1
	file_name = '0'
	while tmp_i < 10000:
		tmp_i *= 10
		file_name += '0'
	file_name += str(i + 1) + '.csv'
	out_file = open(output_dir + file_name, 'w')
	writer = csv.writer(out_file)
	while sample_start + sample_len <= line_len:
		time_data = line[sample_start: sample_start + sample_len]
		freq_data = abs(np.fft.fft(time_data) / sample_len)
		end_data = statistics.mean(freq_data[1:])
		sample_start += 1
		# #########################
		# notice!: time data - freq data - and then mean of freq[1:] in one row
		row_data = np.append(time_data, freq_data)
		row_data = np.append(row_data, end_data)
		writer.writerow(row_data)
	writer.writerow(line[sample_start: line_len])
	out_file.close()
