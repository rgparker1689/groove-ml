import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path


# separates time series into list of segments representing hits
def separate(y, sr):
    # initial data gathering
    hoplength = int(sr/1000)
    margin_adj = hoplength*3
    onsets = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hoplength)
    onsets -= 1.5*np.average(onsets)
    onsets[onsets < 0] = 0
    onsets /= np.max(onsets)
    hpl2 = hoplength
    peaks = librosa.util.peak_pick(onsets, hpl2, hpl2, hpl2, hpl2, .01, hpl2)
    times = librosa.times_like(onsets, sr=sr, hop_length=hoplength)

    # plotting work
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    plt.sca(ax1)
    limits = ax1.get_ylim()
    ax1.plot(times, onsets, label="Onset Strength", alpha=0.6)
    ax1.vlines(times[peaks], 0, onsets.max(), color='r', alpha=0.4, label="Peaks")
    ax1.legend(frameon=True, framealpha=0.8)
    plt.savefig("temp.png")
    plt.close(fig1)

    peaks = np.insert(peaks, 0, 0)
    # building hit audio arrays
    hits = []
    for i in range(peaks.shape[0] - 1):
        start = peaks[i] * hoplength
        finish = (peaks[i+1] + margin_adj) * hoplength
        hit = y[start:finish]
        hits.append(hit)
    return hits


# processes a time series containing a single drum or cymbal hit into a normalized format for interpretation
def preprocess(y, sr, name='default'):
    # begin with search for multiple peaks (invalid input)
    onsets = librosa.onset.onset_strength(y=y, sr=sr, hop_length=16)
    onsets -= 4*np.average(onsets)
    onsets[onsets < 0] = 0
    max_index = np.argmax(onsets) * 16
    nonzero = False
    back_to_zero = False
    mult_peaks = False
    for i in onsets:
        if back_to_zero:
            if i != 0:
                print("Multiple Peaks Detected")
                mult_peaks = True
                break
        if nonzero:
            if i == 0:
                back_to_zero = True
        else:
            if i != 0:
                nonzero = True

    # either crop or elongate to normalize
    if y.shape[0] >= 3000:
        if max_index >= 1499:
            if y.shape[0] - max_index <= 1500:
                end_index = y.shape[0] - 1
                start_index = max_index - (1500 + (1501 - (y.shape[0] - max_index)))
            else:
                start_index = max_index - 1500
                end_index = max_index + 1500
        else:
            start_index = 0
            end_index = max_index + (3000 - max_index)
    else:
        if max_index <= 1500:
            start_index = 0
            y = np.append(np.zeros(1500-max_index), y)
            max_index += (1500 - max_index)
        else:
            start_index = max_index - 1500
        if y.shape[0] - max_index <= 1500:
            y = np.append(y, np.zeros(1500 - (y.shape[0] - max_index)))
            end_index = y.shape[0]
        else:
            end_index = max_index + 1500
        print("start and end indices: ")
        print(start_index, end_index)
    bounds = np.array([start_index, end_index])

    # plotting
    times = librosa.times_like(onsets, sr=sr, hop_length=10)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    plt.sca(ax1)
    ax1.plot(times, onsets, label="Onset Strength", alpha=0.6)
    onset_max = int(max_index/16)
    onset_bounds = (bounds/16).astype(int)
    ax1.vlines(times[onset_max], 0, onsets.max(), color='r', alpha=0.7)
    ax1.vlines(times[onset_bounds], 0, onsets.max(), color='green', alpha=0.7)
    if mult_peaks:
        plt.savefig(name + '.png')
        plt.close(fig1)
        return True, y
    plt.close(fig1)
    y = y[start_index:end_index]
    print(y.shape[0])
    return False, y


y, sr = librosa.load("audios/1_4a120.wav")
temp_hits = separate(y, sr)
total_wrong = 0
for i in temp_hits:
    mult_peaks, y = preprocess(i, sr, str(total_wrong))
    if mult_peaks:
        total_wrong += 1
print(str(total_wrong) + '/' + str(len(temp_hits)))
