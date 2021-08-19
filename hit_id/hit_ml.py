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
dump = True
audio_folder = Path("/Users/rileyparker/PycharmProjects/grooveML/training/")
paths = pd.read_csv('hit_id_training.csv')
paths.fillna('', inplace=True)
labels = list(paths.columns)
data = pd.DataFrame(columns=['Class', 'MFCC', 'SpecBW'])
for i in labels:
    for j in paths[i]:
        print(j)
        if j == '':
            continue
        path = audio_folder/j
        y, sr = librosa.load(path=path)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, hop_length=64, n_bands=6)
        if dump:
            fig, ax = plt.subplots(nrows=1)
            img1 = librosa.display.specshow(spectral_contrast, x_axis='time', ax=ax)
            fig.colorbar(img1, ax=ax)
            ax.set(title='spectral_contrast: ' + j)
            plt.savefig('imgs/spectral_contrast: ' + j + '.png')
            plt.close(fig)

        # MFCC- Solid for differentiating between types of drums/cymbals??
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        fig, ax = plt.subplots(nrows=1)
        img1 = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel',ax=ax)
        fig.colorbar(img1, ax=ax)
        if dump:
            fig, ax = plt.subplots(nrows=1)
            img1 = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img1, ax=ax)
            plt.savefig('imgs/mel_spec: ' + j + '.png')
            plt.close(fig)

        # Spectral Bandwidth- Looks like it may be the key to deciding whether or not a cymbal is also present
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        times = librosa.times_like(spec_bw)
        if dump:
            fig, ax = plt.subplots(nrows=1)
            ax.semilogy(times, spec_bw[0], label='spectral bandwidth')
            ax.set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
            ax.legend()
            ax.label_outer()
            plt.savefig('imgs/spec_bandwidth: ' + j + '.png')
            plt.close(fig)

        data.loc[len(data.index)] = [i, mel_spec, spec_bw]

print(data['MFCC'][0].shape, data['SpecBW'][0].shape)
