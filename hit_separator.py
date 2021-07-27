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
    print(peaks)

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
    onsets = onsets
    times = librosa.times_like(onsets, sr=sr, hop_length=10)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    plt.sca(ax1)
    ax1.plot(times, onsets, label="Onset Strength", alpha=0.6)
    if mult_peaks:
        plt.savefig(name+'.png')
        return True
    plt.close(fig1)
    return False


y, sr = librosa.load("audios/1_2rc100.wav")
temp_hits = separate(y, sr)
total_wrong = 0
for i in temp_hits:
    mult_peaks = preprocess(i, sr, str(total_wrong))
    if mult_peaks:
        total_wrong += 1
print(str(total_wrong) + '/' + str(len(temp_hits)))
