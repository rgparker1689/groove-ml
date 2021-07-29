import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import hit_separator as hs


def beat_placement(path, bpm=0):
    beat_indices = []
    y, sr = librosa.load(path)
    hoplength = int(sr/1000)
    if bpm == 0:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hoplength)
        bpm = tempo
        beat_indices = beats*hoplength
    else:
        bps = bpm/60
        bpf = bps/sr
        spb = 1/bpf
        beat_indices.append(0)
        samp_curr = spb
        while samp_curr <= y.shape[0]:
            beat_indices.append(int(samp_curr))
            samp_curr += spb
    return beat_indices

