import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import hit_separator as hs
from pathlib import Path

# temp source while gathering data for training
# hits = hs.get_hits("audios/1_1a120.wav")[0]
audio_folder = Path("/Users/rileyparker/PycharmProjects/grooveML/training/")
paths = pd.read_csv('hit_id_training.csv')
paths.fillna('', inplace=True)
labels = list(paths.columns)
print(labels)
for i in labels:
    for j in paths[i]:
        print(j)
        if j == '':
            continue
        path = audio_folder/j
        y, sr = librosa.load(path=path)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, hop_length=64, n_bands=6)
        fig, ax = plt.subplots(nrows=1)
        img1 = librosa.display.specshow(spectral_contrast, x_axis='time', ax=ax)
        fig.colorbar(img1, ax=ax)
        ax.set(title='spectral_contrast: ' + j)
        plt.savefig('imgs/spectral_contrast: ' + j + '.png')
        plt.close(fig)




