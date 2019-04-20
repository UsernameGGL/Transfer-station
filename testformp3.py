from pydub import AudioSegment
import numpy as np
import pylab as pl
import wave



song = AudioSegment.from_wav("./prodAudios/00001.wav")
print("len",len(song))
print("frame_rate",song.frame_rate)
print("frame_width",song.frame_width)
print("sample_width",song.sample_width)
print("channels",song.channels)
print('loudness',song.rms)
print('frame_count', song.frame_count())
print('len(raw_data)', len(song.raw_data))
song = song[0:10]
print(song.raw_data)
print(song.frame_rate)
print(len(song))
print(len(song.raw_data))

raw_data = [song.raw_data[i] for i in range(len(song.raw_data))]
audio_out_data = np.array([raw_data[0: 10]])
print(audio_out_data)
audio_out_data = np.append(audio_out_data, [raw_data[10: 20]], axis=0)
print(audio_out_data)

# wf = wave.open("./prodAudios/0.wav")
# print("channels", wf.getnchannels())
# print("frames", wf.getnframes())
# print("frame_rate",wf.getframerate())
# print("sample_width", wf.getsampwidth())

# print(wf.readframes(0))
# print(song.raw_data[3381])
# print(len(song.raw_data))

# print(song.raw_data)
# print(wf.readframes(wf.getnframes()).equals(song.raw_data))

#with open('records-ggl.txt','a') as f:
#	f.write('__dict__:\n{}\nraw_data:\n{}'.format(song.__dict__,song.raw_data))

# print('frame_count',song.frame_count())
# print('frame_count(200)',song.frame_count(ms=1000))

# with open('1.wav', 'wb') as f:
# 	song[1000:6000].export(f, format='wav')


# one_slice = AudioSegment.from_wav("./prodAudios/0.wav")
# fc=one_slice.frame_count()
# print(fc)
# l=len(one_slice.raw_data)
# print(l/fc)
# print(l)
# print(one_slice.raw_data[:int(l/fc)])
# print(one_slice.sample_width)
# time_data = [one_slice.raw_data[i] for i in range(l)]
# time_data = np.array(time_data)
# time_data.shape = -1, 2
# for i in range(len(time_data)):
# 	time_data[i][0] = time_data[i][0]*16*16 + time_data[i][1]
# time_data = time_data.T
# time_data = time_data[0][:]
# fft_size = 1
# tmp_l = l
# while int(tmp_l / 2) != 0:
# 	fft_size *= 2
# 	tmp_l = int(tmp_l / 2)
# # sample_num是2的17次方
# fft_size = int(fft_size / 2)
# #fft_size = 44100
# fft_size = len(time_data)
# fft_data = np.fft.rfft(time_data[:fft_size])/fft_size
# sample_rate = one_slice.frame_rate
# print(sample_rate)
# #freqs = np.linspace(0, sample_rate/2, fft_size/2+1)
# freqs = np.linspace(0, sample_rate/2, fft_size/2+1)

# pl.figure(figsize=(8, 4))
# # pl.subplot(211)
# # pl.plot(t[:fft_size], xs)
# # pl.xlabel(u"时间(秒)")
# # pl.title(u"156.25Hz和234.375Hz的波形和频谱")
# # pl.subplot(212)

# #pl.plot(freqs[1:], fft_data[1:])
# pl.plot(freqs[1:], abs(fft_data[1:]))
# pl.xlabel(u"频率(Hz)")
# pl.show()

# aaa = [[]]
# aaa[0][:] = one_slice[0: 1000]
# aaa.append(one_slice[1001:1500])
# print(aaa)

# print(one_slice.raw_data)

