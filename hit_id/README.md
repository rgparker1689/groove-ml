# hit-id
A machine learning model to predict whether a supplied drum or cymbal audio file/sample
(in the form of a .wav file) is a drum or cymbal, and which type of drum or cymbal it is. 

As of now this project consists of in-progress python scripts which separates a drum groove audio
file into a numpy array containing time series representing drum or cymbal hits, each cropped or
elongated to be the same size with the bounds centered around the peak onset strength when
possible.

The idea is to use this data to train a model which can determine the type of drum or cymbal (as well as whether it is the same tom or cymbal in the common case of a set containing multiple varyingly pitched toms and crashes) for use in a groove-graphing system.
