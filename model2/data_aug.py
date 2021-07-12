import librosa
import matplotlib
import numpy as np
import random

def augment_data(y, sr):
    new_series = []
    # pitch shift
    for i in range(4):
        rand = random.uniform(-12, 12)
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=rand, bins_per_octave=36)
        new_series.append([y_shifted, 'pitched_by_{}_steps'.format(rand)])
    # tempo shift
    for j in range(4):
        rand = random.uniform(0.8, 1.2)
        y_shifted = librosa.effects.time_stretch(y=y, rate=rand)
        new_series.append([y_shifted, 'shifted_by_rate_{}'.format(rand)])
    return new_series
