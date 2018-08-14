import os

import numpy as np
import soundfile as sf
from matplotlib import cm
from matplotlib.image import imsave
from scipy.signal import spectrogram, stft

data_folder = os.path.join('..', '..', 'data', 'recordings')
images_folder = os.path.join('..', '..', 'data', 'images')
albums = os.listdir(data_folder)
recordings = [os.listdir(af) for af in [os.path.join(data_folder, a) for a in albums]]
clip_time = 20
frequency_cap = 5_000
np.random.seed(18)

test = True

if test:
    audio_file = 'audio.flac'
    samples, sample_rate = sf.read(audio_file)
    samples = np.average(samples, axis=1)  # magically transform the file to mono
    # frequencies, times, values = spectrogram(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048)
    f1, t1, v1 = np.abs(stft(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048))
    f2, t2, v2 = np.abs(stft(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048))
    v2 = v2 ** 2
    f3, t3, v3 = spectrogram(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048)
    f4, t4, v4 = np.abs(stft(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048))
    v4 = np.log10(v4 + 1e-3)
    f_end = np.searchsorted(f1, frequency_cap)
    clip_size = int(clip_time / (t1[1] - t1[0]))
    n_clips = int(t1[-1] / clip_time)
    for k in range(n_clips):
        clip_start = np.random.randint(0, len(t1) - clip_size)
        clip_end = clip_start + clip_size
        output_file = os.path.join('test4_{}.png'.format(clip_start))
        s = v4[:f_end, clip_start:clip_end]
        imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
else:
    for i, a in enumerate(albums):
        print("Working on {}".format(a))
        for j, r in enumerate(recordings[i]):
            audio_file = os.path.join(data_folder, a, r)
            samples, sample_rate = sf.read(audio_file)
            samples = np.average(samples, axis=1)  # magically transform the file to mono
            f1, t1, v1 = np.abs(stft(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048))
            f_end = np.searchsorted(f1, frequency_cap)
            clip_size = int(clip_time / (t1[1] - t1[0]))
            n_clips = int(t1[-1] / clip_time)
            for k in range(n_clips):
                clip_start = np.random.randint(0, len(t1) - clip_size)
                clip_end = clip_start + clip_size
                output_file = os.path.join(images_folder, 'CRS_{}_{}_{}.png'.format(i, j, clip_start))
                s = v1[:f_end, clip_start:clip_end]
                imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
