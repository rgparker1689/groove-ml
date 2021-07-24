# groove-ml
A machine learning model to predict whether a supplied drum groove/loop/audio file is programmed (i.e. MIDI/Drum Machines) or performed.

A number of audio features have been tested. Began with tempograms and fourier tempograms, but found very little success with well performed drum parts.
Would certainly need audio lengths in excess of 30 seconds to find much use here. I suspect the biggest reason for this is the incredibly small hop length
(how many samples are skipped) required to ensure that the onset strength envelope is determined with sufficient precision to be a helpful.

In the second model, which reached 100% accuracy given limited MIDI dynamics (this will be challenged soon, though what I believe to be the best feature is
independent of dynamics), I approached the onset strength envelope again. I calculated the tempogram manually to use a lower hop length for the onset
envelope -- to satisfy my curiosity mostly, I didn't use it as a feature -- by turning the envelope into a self-similarity matrix. I then smoothed it with
librosa's diagonal filters, which seemed to help significantly. 

I also extracted the spectral flatness time series of the grooves. This is essentially how noise-like a sound is as opposed to tonal. I was wary of 
overfitting here, so I am using a number of different drum sets both for samples and for performances. Most issues would be dealt with by turning it
into an SSM, though. After creating the SSM I smoothed it like the other and ran both through a fairly straightforward CNN for classification.
