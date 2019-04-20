import csv
from pydub import AudioSegment
import os
import numpy as np

path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'prodAudios/'
labelPath = path+'prodLabels.csv'

audioDir = os.listdir(audioPath)

labelFile = open(labelPath)

labelReader = csv.reader(labelFile)
label_input = list(labelReader)

twoChannelPicDir = './twoChannelPic/'
if not os.path.exists(twoChannelPicDir):
	os.makedirs(twoChannelPicDir)

label_index = 0 
first_la_output = True
for audioFName in audioDir:
	print(audioFName)
	audioFile = AudioSegment.from_wav(audioPath+audioFName)
	if audioFile.channels != 1:
		print('error! channel num is larger than 1!')
		os._exit(0)
	raw_data = [audioFile.raw_data[i] for i in range(len(audioFile.raw_data))]
	raw_data = np.array(raw_data)
	raw_data.shape = -1, 2
	for i in range(len(raw_data)):
		raw_data[i][0] = raw_data[i][0]*16*16 + raw_data[i][1]
	raw_data = raw_data.T
	raw_data = raw_data[0][:]
	sample_count = 128*128
	start = 0
	frame_count = audioFile.frame_count()
	while frame_count - sample_count >= 0:
		if first_la_output:
			first_la_output = False
			label_output = [label_input[0]]
			audio_out_data = np.array([raw_data[start: start+sample_count]])
		else:
			label_output.append(label_input[label_index])
			np.append(audio_out_data, [raw_data[start: start+sample_count]], axis=0)
		frame_count -= sample_count
		start += sample_count
	label_index += 2
	#print('处理完第{}个文件了'.format(label_index/2+1))

print('把音频写入文件')
np.savetxt('twoChannelPic.txt', 
			np.array(audio_out_data))

print('将label写入文件')
laOutFName = './twoChannelLabel.csv'
laExportFile = open(laOutFName, 'w')
laExportWriter = csv.writer(label_output)
for item in label_output:
	laExportWriter.writerow(item)
laExportFile.close()


