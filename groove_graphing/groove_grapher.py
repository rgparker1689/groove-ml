import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa


def beat_placement(path, bpm=0):
    beat_indices = []
    y, sr = librosa.load(path)
    hoplength = int(sr/1000)
    if bpm == 0:  # No bpm supplied, infer from peaks
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hoplength)
        bpm = tempo
        beat_indices = beats*hoplength
    else:  # TODO: Replace beat centering with superior logic to give most leeway.
        bps = bpm/60
        bpf = bps/sr
        spb = 1/bpf
        beat_indices.append(0)
        samp_curr = spb
        while samp_curr <= y.shape[0]:
            beat_indices.append(int(samp_curr))
            samp_curr += spb
    return np.divide(beat_indices, 22050, casting='unsafe')


def graph(types, indices, save=False, beat_path='none', bpm=0):
    print(types)
    print(indices)
    heights = {'Kick': 1, 'Floor Tom': 2, 'Snare': 3, 'High Tom': 5, 'Mid Tom': 4, 'Hi-Hat/Ride': 7, 'Crash': 8}
    adj_loc = np.divide(indices, 22050, casting='unsafe')
    hit_y = [heights[i] for i in types]
    plt.scatter(adj_loc, hit_y, color='r')
    plt.ylim([0, 9])
    plt.hlines(np.arange(9), 0, max(adj_loc), linestyles='--', alpha=0.7)
    if beat_path != 'none':
        plt.vlines(beat_placement(beat_path, bpm=bpm), 0, 9)
    plt.yticks([1, 2, 3, 4, 5, 7, 8],['Kick', 'Floor Tom', 'Snare', 'Mid Tom', 'High Tom', 'Hi-Hat/Ride', 'Crash'])
    plt.title('Groove Plot:')
    plt.xlabel('Time (s)')
    plt.show()
    if save:
        plt.savefig('graph.png')
