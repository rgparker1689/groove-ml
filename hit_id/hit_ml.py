import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import hit_separator as hs

# temp source while gathering data for training
hits = hs.get_hits("audios/1_1a120.wav")[0]
data = []
for idx, i in enumerate(hits):
    spectral_contrast = librosa.feature.spectral_contrast(y=i, hop_length=64, n_bands=6)
    fig, ax = plt.subplots(nrows=1)
    img1 = librosa.display.specshow(spectral_contrast, x_axis='time', ax=ax)
    fig.colorbar(img1, ax=ax)
    ax.set(title='spectral_contrast: ' + str(idx))
    plt.savefig('spectral_contrast: ' + str(idx))
    plt.close(fig)

