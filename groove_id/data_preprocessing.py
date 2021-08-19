import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow.keras as tf
from sklearn.model_selection import train_test_split

def groove_preprocess(df, bsize):

    x1 = np.array(df[df.columns[0]].tolist())
    x2 = np.array(df[df.columns[1]].tolist())
    x3 = np.array(df[df.columns[2]].tolist())
    x4 = np.array(df[df.columns[3]].tolist())
    y = np.array(df[df.columns[4]].tolist())
    print(y)

    le = LabelEncoder()
    yy = tf.utils.to_categorical(le.fit_transform(y))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(x2.shape)
    print(x2)
    scaler.fit_transform(x1)

    # come back to, serious flaw
    for i in x2:
        scaler.fit_transform(i)
    for i in x3:
        scaler.fit_transform(i)
    for i in x4:
        scaler.fit_transform(i)

    x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, x4_train, x4_test, y_train, y_test = train_test_split(x1, x2, x3, x4, yy, test_size=0.33, random_state=42)
    train = [x1_train, x2_train, x3_train, x4_train, y_train]
    test = [x1_test, x2_test, x3_test, x4_test, y_test]
    return train, test

def sample_preprocess(df, bsize):

    # separating into ndarrays
    labels = df['Class'].to_numpy()
    mels = np.array(df['MFCC'].tolist())
    bandw = np.array(df['SpecBW'].tolist())
    flatn = np.array(df['SpecFL'].tolist())

    # min/max scaling (0, 1)
    mels /= np.amax(mels)
    bandw /= np.amax(bandw)
    flatn /= np.amax(flatn)

    # ordinal encoding for classes
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    scaler = MinMaxScaler()

    x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(mels, bandw, flatn, labels, test_size=0.33, random_state=42)
    train = [x1_train, x2_train, x3_train, y_train]
    test = [x1_test, x2_test, x3_test, y_test]
    return train, test
