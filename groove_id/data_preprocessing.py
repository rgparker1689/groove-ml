import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow.keras as tf
from sklearn.model_selection import train_test_split

def preprocess(df, bsize):

    x1 = np.array(df[df.columns[0]].tolist())
    x2 = np.array(df[df.columns[1]].tolist())
    x3 = np.array(df[df.columns[2]].tolist())
    x4 = np.array(df[df.columns[3]].tolist())
    y = np.array(df[df.columns[4]].tolist())

    le = LabelEncoder()
    yy = tf.utils.to_categorical(le.fit_transform(y))

    # scaler = MinMaxScaler(feature_range=(0,1))
    # x1 = scaler.fit_transform(x1)
    # x2 = scaler.fit_transform(x2)
    # x3 = scaler.fit_transform(x3)
    # x4 = scaler.fit_transform(x4)

    x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, x4_train, x4_test, y_train, y_test = train_test_split(x1, x2, x3, x4, yy, test_size=0.33, random_state = 42)
    train = [x1_train, x2_train, x3_train, x4_train, y_train]
    test = [x1_test, x2_test, x3_test, x4_test, y_test]
    return train, test
