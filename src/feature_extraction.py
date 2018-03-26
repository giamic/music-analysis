"""
In this file, we shall take as an input an audio recording, sample randomly 10 seconds out of it, then extract features
from it using Essentia. Tentatively, the features to extract in the first moment are the FFT coefficients.
"""

import numpy as np
import soundfile as sf


def sample_excerpt(audio, seconds, fs=44100):
    """
    Take a long audio track and return a short excerpt lasting for 'seconds'.

    :param audio: a wav audio track
    :param seconds: the length in seconds of the excerpt
    :param fs: the sampling frequency
    :return: an excerpt from audio of the correct duration
    """
    length = seconds * fs
    init = np.random.random() * len(audio) - length  # make sure you don't start

    return audio[init:init + length]


sf.read("data/Bach;_Glenn_Gould-Goldberg_Variations,_BWV_988/01.Partita_for_keyboard_no._5_in_G_major,_BWV_829_I._Praeambulum.flac")