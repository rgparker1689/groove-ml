from comet_ml import Experiment
import keras.losses
import tensorflow
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as tf
from tensorflow.python.framework.ops import disable_eager_execution
from pathlib import Path
from data_aug import *
from data_preprocessing import *


audio_folder = Path("/Users/rileyparker/PycharmProjects/grooveML/audios/")
names = pd.read_csv('audiowavstemp.csv')
labels = list(names.columns)
audiowavs = dict()
for i in labels:
    s = names[i].dropna()
    audiowavs[i] = s

# gather and augment data
data = []
for label in audiowavs:
    for i in audiowavs[label]:
        y, sr = librosa.load(path=audio_folder / i, duration=11)
        augmented = augment_data(y, sr)
        temp = [[y[:180000], sr, i]]
        for obj in augmented:
            temp.append([obj[0][:180000], sr, i[:-4] + obj[1]])
        for j in temp:
            # reusable variables
            print('operating on: ' + (audio_folder / j[2]).stem)
            temp = audio_folder / j[2]
            hoplength = 2048

            onset_lowhop = librosa.onset.onset_strength(y=j[0], sr=j[1], hop_length=64)
            onset = librosa.onset.onset_strength(y=j[0], sr=j[1], hop_length=hoplength)
            print("Onset complete.")
            peaks = librosa.util.peak_pick(onset_lowhop, 3, 3, 3, 5, 0.03, 0)
            peak_dists = []
            for i in range(peaks.shape[0]):
                if i > 0:
                    peak_dists.append(peaks[i]-peaks[i-1])
            times = librosa.times_like(onset_lowhop, sr=j[1], hop_length=64)

            # plotting peaks
            fig2 = plt.figure()
            ax1 = fig2.add_subplot()
            plt.sca(ax1)
            ax1.plot(times, onset_lowhop, label='Onset Strength', alpha=0.6)
            ax1.vlines(times[peaks], 0, onset_lowhop.max(), color='r', alpha=0.4, label='Peaks')
            ax1.legend(frameon=True, framealpha=0.8)
            name = temp.stem + '_peaks.png'
            print(name)
            plt.savefig(temp.stem + '_peaks.png')
            plt.close(fig2)
            print(int(j[0].shape[0]/hoplength))

            print("Starting ssms:")
            self_recmat = librosa.segment.recurrence_matrix(data=onset, mode='connectivity', self=False, width=5)
            print("connectivity complete")
            self_affmat = librosa.segment.recurrence_matrix(data=onset, mode='affinity', self=False, width=5)
            print("affinity complete")
            affmat_smoothed = librosa.segment.path_enhance(R=self_affmat, n=51, window='hann', n_filters=7, zero_mean=True)
            print("affinity smoothed")

            # plotting ssms
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
            imgsim = librosa.display.specshow(self_recmat, x_axis='s', y_axis='s',
                                              hop_length=hoplength, ax=ax[0])
            ax[0].set(title='Binary')
            imgaff = librosa.display.specshow(self_affmat, x_axis='s', y_axis='s',
                                              hop_length=hoplength, cmap='magma_r', ax=ax[1])
            ax[1].set(title='Affinity')
            ax[1].label_outer()
            imgsmo = librosa.display.specshow(affmat_smoothed, x_axis='s', y_axis='s',
                                              hop_length=hoplength, cmap='magma_r', ax=ax[2])
            ax[2].set(title='Smoothed Affinity')
            ax[2].label_outer()
            fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
            fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
            fig.colorbar(imgsmo, ax=ax[2], orientation='horizontal', ticks=[0, 1])
            plt.savefig(temp.stem)
            plt.close(fig)
            print("plotting complete")

            # gathering other features
            flatness = librosa.feature.spectral_flatness(y=j[0])
            flatness_mat = librosa.segment.recurrence_matrix(data=flatness, mode='affinity', self=False, width=5)
            zerox_rate = librosa.feature.zero_crossing_rate(y=onset, frame_length=220500, hop_length=512)
            num_zerox = zerox_rate[0, zerox_rate.shape[1] - 1] * zerox_rate.shape[1]
            print("flatness complete")

            # plotting flatness
            fig3 = plt.figure()
            ax2 = fig3.add_subplot()
            plt.sca(ax2)
            imgfla = librosa.display.specshow(flatness_mat, x_axis='s', y_axis='s',
                                              hop_length=hoplength, cmap='magma_r', ax=ax2)
            fig3.colorbar(imgfla, ax=ax2, orientation='horizontal')
            plt.savefig(temp.stem + '_flatness.png')
            plt.close(fig3)

            # appending data
            data.append([onset_lowhop, self_recmat, affmat_smoothed, flatness_mat, label])

data_df = pd.DataFrame(data, columns=['onset', 'recurrence', 'affinity', 'flatness', 'label'])
train, test = preprocess(data_df, bsize=2)
