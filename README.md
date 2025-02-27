# ARMAS: Active Reconstruction of Missing Audio Segments

## Description

Signal degradation due to noise is a common challenge in various engineering applications. This project focuses on reconstructing missing or corrupted parts of signals using advanced interpolation and estimation techniques to fill in gaps and restore the signal's integrity. Different Machine learning techniques including Simple RNN, LSTM, SVR, GRU, are used to train and predict missing values. The effectiveness of these models is evaluated using FFT, MSE, Spectral Centroid and correlation.

## Results

The image below shows the reconstruction of lost audio signals with four different techniques: Steganoflage, Random Forest, SVR (Support Vector Regressor) and LSTM (Long Short Term Memory).

![alt text](\OutAudioplots\pop11 reconstructed.png)

## Demo Audio

### Original Audio
https://user-images.githubusercontent.com/91117093/205602462-c61c9db8-b571-416c-9073-40c5c2eb0f68.mp4

### Dropped Audio
https://user-images.githubusercontent.com/91117093/205603165-438346c2-8824-44a6-8857-a8e4c87fef99.mp4

### Stego Reconstruction
https://user-images.githubusercontent.com/91117093/205603236-12e2026a-d170-4694-8e95-b5247d40986a.mp4

### Random Forest Reconstruction
https://user-images.githubusercontent.com/91117093/205603257-61d89bdc-a8b8-4d5a-a011-ef72fbda0227.mp4

### SVR Reconstruction
https://user-images.githubusercontent.com/91117093/205603303-6884b1a0-76d4-4bc0-9206-7e5bdb9ee67b.mp4

### LSTM Reconstruction
https://user-images.githubusercontent.com/91117093/205603364-32e6abaf-3a80-49b2-a77c-e9eb7bbb340d.mp4

