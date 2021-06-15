from comet_ml import Experiment
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

disable_eager_execution()

# comet_ml tracking wrapper
experiment = Experiment(
    api_key="F9dUYytBwvR290XYycBVCeJ5T",
    project_name="groove_mlAW",
    workspace="rgparker1689",
)

# reads audio spreadsheet into dataframe and creates dict mapping labels to series with their files\
audio_folder = Path("C:/Users/rgpar/PycharmProjects/groove_mlAW/audios/")
names = pd.read_csv('audiowavs.csv')
labels = list(names.columns)
audiowavs = dict()
for i in labels:
    s = names[i].dropna()
    audiowavs[i] = s

# outputs waveforms to comet
wvfrms = plt.figure(figsize=(15, 15))
experiment.log_image('waveforms.png')
wvfrms.subplots_adjust(hspace=0.4, wspace=0.4)
index_counter = 0  # prevents overlap on png
for i, label in enumerate(labels):
    for j in range(audiowavs[label].size):
        func = audio_folder / audiowavs[label].loc[j]
        wvfrms.add_subplot(10, 2, index_counter + 1)
        plt.title(label + ": " + func.stem)
        y, sample_rate = librosa.load(func)
        librosa.display.waveplot(y, sr=sample_rate)
        index_counter += 1
        experiment.log_audio(func, metadata={'name': label})  # logs wav
plt.savefig('waveforms.png')
experiment.log_image('waveforms.png')

# outputs tempogram/bpm estimate graphs to comet
prior = scipy.stats.uniform(30, 300)  # ASSUMES BPM IN 30,300
tmpgrms = plt.figure(figsize=(15, 15))
experiment.log_image('tempograms.png')
tmpgrms.subplots_adjust(hspace=1.4, wspace=0.4)
index_counter = 0
for i, label in enumerate(labels):
    for j in range(audiowavs[label].size):
        func = audio_folder / audiowavs[label].loc[j]
        tmpgrms.add_subplot(10, 2, index_counter + 1)
        y, sample_rate = librosa.load(func)
        hop = 256
        oenv = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop)
        tempogram = librosa.feature.tempogram(y=y, onset_envelope=oenv, sr=sample_rate, hop_length=hop)
        tempo_estimate = librosa.beat.tempo(y=y, onset_envelope=oenv, sr=sample_rate, prior=prior)
        librosa.display.specshow(data=tempogram, sr=sample_rate, hop_length=hop, x_axis='time', y_axis='tempo')
        plt.title(label + ": " + str(tempo_estimate) + " " + func.stem)
        index_counter += 1
plt.savefig('tempograms.png')
experiment.log_image('tempograms.png')

# outputs autocorrelation graphs to comet
acrlts = plt.figure(figsize=(15, 15))
experiment.log_image('autocorrelations.png')
acrlts.subplots_adjust(hspace=1.4, wspace=0.4)
index_counter = 0
for i, label in enumerate(labels):
    for j in range(audiowavs[label].size):
        func = audio_folder / audiowavs[label].loc[j]
        acrlts.add_subplot(10, 2, index_counter + 1)
        y, sample_rate = librosa.load(func)
        hop = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop)
        ac = librosa.autocorrelate(oenv, max_size=4 * sample_rate / 512)
        plt.plot(ac)
        plt.title(label + ": autocorrelation " + func.stem)
        plt.xlabel('Lag (frames)')
        index_counter += 1
plt.savefig('autocorrelations.png')
experiment.log_image('autocorrelations.png')


# function for gathering local and global onset autocorrelations for each file
def gather_autocor(path, hop):
    y, sr = librosa.load(path=path, duration=3)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop) # local autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0]) # global autocorrelation
    return tempogram, ac_global


# add additional feature functions here

# creating a list of respective autocorrelations and converting to df
features = []
for label in labels:
    for i in audiowavs[label]:
        local_ac, global_ac = gather_autocor(audio_folder / i, 256)
        features.append([local_ac, global_ac, label])
featuresdf = pd.DataFrame(features, columns=['local', 'global', 'label'])

# preprocessing
x1 = np.mean(featuresdf['local'].tolist(), axis=1)  # averaging local acs for each audio file to match shape of global
x2 = np.array(featuresdf['global'].tolist()) # reformatting global array into manageable shape ^
y = featuresdf['label'].tolist()
le = LabelEncoder()
yy = tf.utils.to_categorical(le.fit_transform(y))
np.savetxt("x1.csv", x1, delimiter=",")
np.savetxt("x2.csv", x2, delimiter=",")
scaler = MinMaxScaler(feature_range= (0, 1))
x1 = scaler.fit_transform(x1)
x2 = scaler.fit_transform(x2)
np.savetxt("x1fitted.csv", x1, delimiter=",")
np.savetxt("x2fitted.csv", x2, delimiter=",", fmt='%s')

# setting up training/testing sets
batch_size = 5
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, yy, test_size=0.33, random_state=42)
np.savetxt("x1train.csv", x1_train, delimiter=",")
np.savetxt("x2train.csv", x2_train, delimiter=",")
np.savetxt("x1test.csv", x1_test, delimiter=",")
np.savetxt("x2test.csv", x2_test, delimiter=",")
np.savetxt("ytrain.csv", y_train, delimiter=",")
np.savetxt("ytest.csv", y_test, delimiter=",")

# Working on model here
input_shape = (259, 1)  # 259 timesteps, 1 feature at each step (batch sized sequences left out until fit call?)
lstm = tf.layers.LSTM(4)
input1 = tf.Input(shape=(259, 1))  # local ac
x1 = input1
input2 = tf.Input(shape=(259, 1))  # global ac
x2 = input2
x1 = lstm(x1)
x2 = lstm(x2)
merged = tf.layers.Concatenate(axis=1)([x1, x2])  # concatenating dim 4 lstm outputs
prediction = tf.layers.Dense(8, activation='relu')(merged)
prediction = tf.layers.Dense(1, activation='sigmoid')(prediction)  # final prediction output 0-1
model = tf.Model(
    inputs=[input1, input2],
    outputs=[prediction],
)
model.summary()
