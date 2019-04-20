import csv
from pydub import AudioSegment
import os

path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'CAL500_32kps/'
labelPath = path+'SegLabelHard/'

audioDir = os.listdir(audioPath)
labelDir = os.listdir(labelPath)

i = 0
labelExport = [[0]]
audioExport = [[0]]
exportNum = 0
auProDir = './prodAudios/'
if not os.path.exists(auProDir):
	os.makedirs(auProDir)
for audioFName in audioDir:
	auPrefix = audioFName.split('.')[0]
	laPrefix = labelDir[i].split('.')[0]
	if auPrefix != laPrefix:
		print('error! files not in relevant.')
		os._exit(0)
	audioFile = AudioSegment.from_mp3(audioPath+audioFName)
	labelFile = open(labelPath+labelDir[i])
	labelReader = csv.reader(labelFile)
	labels = list(labelReader)
	for j in range(1, len(labels)):
		start = float(labels[j][0])*1000
		end = float(labels[j][1])*1000
		if start == end:
			print(audioFName+'的第'+str(j)+'段有问题，start==end')
		elif start > end:
			print(audioFName+'的第'+str(j)+'段有问题，start>end')
		for k in range(2, 20):
			#print(j,k)
			if (i == 0 and j == 1 and k == 2):
				labelExport[0][0] = int(float(labels[j][k]))
			elif k == 2:
				labelExport.append([labels[j][k]])
			else:
				labelExport[exportNum].append(int(float(labels[j][k])))
		
		# if i == 0 and j == 1:
		# 	audioExport[0][:] = audioFile[start: end]
		# else:
		# 	audioExport.append(audioFile[start: end])

		# with open('{}{}.wav'.format(auProDir, exportNum), 'wb') as f:
		# 	audioFile[start: end].export(f, format='wav')
		exportNum += 1
		# labelExport.append([])
	i += 1
	#audioFile.close()
	labelFile.close()
	#print('已经处理完了{}乘以2个文件诶'.format(i))

# print('下面该写label了噢')

# laProFName = './prodLabels_1.csv'
# laExportFile = open(laProFName, 'w')
# laExportWriter = csv.writer(laExportFile)
# for item in labelExport:
# 	if item != []:
# 		laExportWriter.writerow(item)
# laExportFile.close()


# i=0
# for audioFName in audioDir:
# 	if audioFName.split('.')[0]!=labelDir[i].split('.')[0]:
# 		print(1)
# 	i += 1


#print(audioDir[0].split('.')[0],audioDir[0].split('.')[1])
#print(labelDir[0].split('.'))
#for audioName in audioNames:

