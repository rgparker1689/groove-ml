import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from pathlib import Path
from groove_id.data_preprocessing import sample_preprocess


def random_classify(hits):
    classes = ['Kick', 'Floor Tom', 'Snare', 'Mid Tom', 'High Tom', 'Hi-Hat/Ride', 'Crash']
    return [classes[random.randint(0, 6)] for i in hits]


def assemble(hits):
    data = pd.DataFrame(columns=['MFCC', 'SpecBW', 'SpecFL'])
    for idx, i in enumerate(hits):
        mel_spec = librosa.feature.melspectrogram(y=i, sr=22050)
        spec_bw = librosa.feature.spectral_bandwidth(y=i, sr=22050, hop_length=128)
        spec_flatness = librosa.feature.spectral_flatness(y=i, hop_length=128)
        data.loc[idx] = [mel_spec, spec_bw, spec_flatness]
    print(data.shape)
    return data


# temp source while gathering data for training
# hits = hs.get_hits("audios/1_1a120.wav")[0]

if __name__ == '__main__':  # Training Model
    dump = False
    audio_folder = Path('/Users/rileyparker/PycharmProjects/grooveML/training/')
    paths = pd.read_csv('/Users/rileyparker/PycharmProjects/grooveML/hit_id_training.csv')
    paths.fillna('', inplace=True)
    labels = list(paths.columns)
    data = pd.DataFrame(columns=['Class', 'MFCC', 'SpecBW', 'SpecFL'])
    for i in labels:
        for j in paths[i]:
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
                plt.savefig('/Users/rileyparker/PycharmProjects/grooveML/imgs/spectral_contrast: ' + j + '.png')
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
                plt.savefig('/Users/rileyparker/PycharmProjects/grooveML/imgs/mel_spec: ' + j + '.png')
                plt.close(fig)

            # Spectral Bandwidth- Looks like it may be the key to deciding whether or not a cymbal is also present
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=128)
            times = librosa.times_like(spec_bw)
            if dump:
                fig, ax = plt.subplots(nrows=1)
                ax.semilogy(times, spec_bw[0], label='spectral bandwidth')
                ax.set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
                ax.legend()
                ax.label_outer()
                plt.savefig('/Users/rileyparker/PycharmProjects/grooveML/imgs/spec_bandwidth: ' + j + '.png')
                plt.close(fig)

            # Spectral Flatness- Trial
            spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=128)
            times = librosa.times_like(spec_flatness)
            if dump:
                fig, ax = plt.subplots(nrows=1)
                ax.semilogy(times, spec_flatness[0], label='spectral flatness')
                ax.set(xticks=[], xlim=[times.min(), times.max()])
                ax.legend()
                ax.label_outer
                plt.savefig('/Users/rileyparker/PycharmProjects/grooveML/imgs/spec_flatness ' + j + '.png')
                plt.close(fig)

            data.loc[len(data.index)] = [i, mel_spec, spec_bw, spec_flatness]

    train, test = sample_preprocess(data, bsize=5)



