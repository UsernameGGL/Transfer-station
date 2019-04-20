from pydub import AudioSegment
import os
path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
input_path = path + 'CAL500_32kps/'
output_path = path + 'CAL500_32kps_wav/'
if not os.path.exists(output_path):
	os.makedirs(output_path)
input_dir = os.listdir(input_path)
for audioFName in input_dir:
	audioFile = AudioSegment.from_mp3(input_path+audioFName)
	with open('{}{}.wav'.format(output_path, audioFName.split('.')[0]), 'wb') as f:
		audioFile.export(f, format='wav')
