import os
path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'prodAudios/'
audioDir = os.listdir(audioPath)
for audioFName in audioDir:
	print('修改前是'+audioFName)
	intFName = int(audioFName.split('.')[0])+1
	tmp = intFName
	element = ''
	while tmp < 10000:
		element += '0'
		tmp *= 10
	os.rename(audioPath+audioFName, audioPath+element+str(intFName)+'.wav')
	print('修改后是'+element+str(intFName)+'.wav')
