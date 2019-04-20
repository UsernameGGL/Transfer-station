import csv
from pydub import AudioSegment
import os
import numpy as np

path = './'
audioPath = path+'CAL500_32kps/'
labelPath = path+'SegLabelHard/'

audioDir = os.listdir(audioPath)
labelDir = os.listdir(labelPath)

i = 0
exportNum = 0
audioOutFile = open('prodAudios_server', 'w')
for audioFName in audioDir:
	auPrefix = audioFName.split('.')[0]
	for ii in range(len(labelDir)):
		laPrefix = labelDir[ii].split('.')[0]
		if(laPrefix == auPrefix):
			labelFName = labelDir[ii]
			break
	if auPrefix != laPrefix:
		print('error! files not in relevant.')
		os._exit(0)
	audioFile = AudioSegment.from_mp3(audioPath+audioFName)
	labelFile = open(labelPath+labelFName)
	labelReader = csv.reader(labelFile)
	labels = list(labelReader)
	for j in range(1, len(labels)):
		start = float(labels[j][0])*1000
		end = float(labels[j][1])*1000
		oneSlice = audioFile[start: end]
		raw_data = [oneSlice.raw_data[i] for i in range(len(oneSlice.raw_data))]
		raw_data = np.array(raw_data)
		raw_data.shape = -1, 2
		for k in range(len(raw_data)):
			raw_data[k][0] = raw_data[k][0]*16*16 + raw_data[k][1]
			audioOutFile.write(str(raw_data[k][0])+' ')
		audioOutFile.write('\n')
		# raw_data = raw_data.T
		# raw_data = raw_data[0][:]
		#print(np.array([raw_data]).shape)
		if i==0 and j == 1:
			labelExport = [list(map(int, list(map(float, labels[j][2:20]))))]
			#audioExport = np.array([raw_data])
			print('laExport j == 1', labelExport)
		else:
			labelExport.append(list(map(int, list(map(float, labels[j][2:20])))))
			if i==0 and j == 2:
				print('laExport j == 2', labelExport)
			#np.append(audioExport, np.array([raw_data]), axis=0)
		
		exportNum += 1
	i += 1
	labelFile.close()
	print('已经处理完了{}乘以2个文件诶'.format(i))

audioOutFile.close()

print('下面该写label_csv了')

laProFName = './prodLabels_server.csv'
laExportFile = open(laProFName, 'w')
laExportWriter = csv.writer(laExportFile)
for item in labelExport:
	if item != []:
		laExportWriter.writerow(item)
laExportFile.close()



