import csv
import os
import numpy as np

out_pic_dir = 'pic_data/'
if not os.path.exists(out_pic_dir):
    os.makedirs(out_pic_dir)
out_lab_dir = 'pic_label/'
if not os.path.exists(out_lab_dir):
	os.makedirs(out_lab_dir)

label_file = open('prodLabels_server.csv', 'r')
label_reader = csv.reader(label_file)
reader_list = list(label_reader)
labels = [[int(reader_list[i][j]) for j in range(len(reader_list[i]))] for i in range(len(reader_list))]


input_file = open('prodAudios_server', 'r')
input_lines = input_file.readlines()
file_name = 0
for line in input_lines:
	line = line.split(' ')
	line.pop()
	line = list(map(int, line))
	line_len = len(line)
	sample_start = 0
	sample_len = 256*256
	with open(out_pic_dir+str(file_name)+'txt', 'a') as pic_file , open(out_lab_dir+str(file_name)+'csv', 'a') as label_file:
		while sample_start + sample_len <= line_len:
			time_data = line[sample_start: sample_start+sample_len]
			freq_data = abs(np.fft.fft(time_data)/sample_len)
			pic_file.write(str(time_data[0]))
			for i in range(1, sample_len):
				pic_file.write(' '+str(time_data[i]))
			for i in range(sample_len):
				pic_file.write(' '+str(freq_data[i]))
			pic_file.write('\n')
			sample_start += 128
		label_file.write(labels[file_name][0])
		for i in range(1, len(labels[file_name])):
			label_file.write(' '+str(labels[file_name][i]))
	file_name += 1

