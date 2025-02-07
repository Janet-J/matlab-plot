from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.io import loadmat
from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from scipy.io.wavfile import write
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import SimpleRNN
from tensorflow.python.keras.layers import LSTMV1
from tensorflow.python.keras.layers import Dropout
# from tensorflow.python.keras.optimizers import SGD
import os


def networks(md, X_train, y_train):

    if md != 'random_forest':
        X_train, y_train = np.array(X_train), np.array(y_train)
      #  X_test = np.array(test_data)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
      #  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    if md == 'lstm':
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True,
                  input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=128)

    elif md == 'simple_rnn':
        model = Sequential()
        model.add(SimpleRNN(128, return_sequences=True,
                  input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(Dense(units=1))  # The time step of the output

        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=0)

    # elif md == 'gru':
    #     # The GRU architecture
    #     model = Sequential()
    #     # First GRU layer with Dropout regularisation
    #     model.add(GRU(units=32, return_sequences=True, input_shape=(
    #         X_train.shape[1], 1), activation='tanh'))
    #     model.add(Dropout(0.2))
    #     # Second GRU layer
    #     model.add(GRU(units=32, return_sequences=True, activation='tanh'))
    #     model.add(Dropout(0.2))

    #     # Third GRU layer
    #     model.add(GRU(units=32, return_sequences=True, activation='tanh'))
    #     model.add(Dropout(0.2))
    #     # Fourth GRU layer
    #     model.add(GRU(units=32, activation='tanh'))
    #     model.add(Dropout(0.2))
    #     # The output layer
    #     model.add(Dense(units=1))
    #     # Compiling the RNN
    #     model.compile(optimizer=SGD(lr=0.01, decay=1e-7,
    #                   momentum=0.6, nesterov=False), loss='mean_squared_error')
    #     # Fitting to the training set
    #     model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)

    elif md == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

    elif md == 'svr':
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples, nx*ny))
        y_train_2d = np.array(y_train)
        y_train_1d = y_train_2d.flatten()

        model = LinearSVR(verbose=0, C=1, epsilon=0, fit_intercept=True,
                          intercept_scaling=1.0, max_iter=1000, random_state=None, tol=0.0001)
        model.fit(X_train_2d, y_train_1d)

    else:
        print('No Such Model')

    return model


def calculations(file_path, model_name):
    original_mat = pd.DataFrame(loadmat(file_path+'original.mat')['y'])
    hidden_mat = pd.DataFrame(loadmat(file_path+'hiddenData.mat')['KK'])
    stego_mat = pd.DataFrame(loadmat(file_path+'stegoAudio.mat')['W'])
    stego_recon = pd.DataFrame(
        loadmat(file_path+'reconstructedAudio.mat')['recon'])

    nan_index = stego_mat[0].index[stego_mat[0].apply(np.isnan)]
    stego_mat.iloc[nan_index] = 0

    original_drop_mat = original_mat.copy()
    original_drop_mat.iloc[nan_index] = 0

    hidden_drop_mat = hidden_mat.copy()
    hidden_drop_mat.iloc[nan_index] = 0

    test_data = hidden_mat.copy()
    test_data = test_data.iloc[nan_index]

    X_train = hidden_drop_mat
    #y_train = original_drop_mat;
    y_train = stego_mat

    # Model Calls
    model = networks(model_name, X_train, y_train)

    X_test = test_data
    plt.plot(X_test)
    X_test = np.array(test_data)
    
    if model_name in ['lstm', 'simple_rnn', 'gru']:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_recon_audio = model.predict(X_test)

    final_mat = stego_mat.copy()
    min_length = min(len(nan_index), len(predicted_recon_audio))

    for i in range(min_length):
        final_mat.iloc[nan_index[i]] = predicted_recon_audio[i]

    signalOut = final_mat.values
    reconstructed_audio = {"reconL": final_mat.values, "Fs": 22050.0}
    sp.savemat(file_path + model_name + "_reconstructed_audio.mat", reconstructed_audio)
    write(file_path + 'OutAudio/reconstructedRF.wav', 22050, signalOut.astype(np.float64))

    recon_mat = pd.DataFrame(final_mat.values)
    frames = pd.concat([original_mat, recon_mat, stego_mat, stego_recon], axis=1)
    frames.columns = ['original', model_name, 'stego_audio', 'stego_recon']
    print(frames.corr())

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    axes[0].plot(original_mat, color='blue', label='Original Audio')
    axes[0].set_title('Original Audio')
    axes[1].plot(recon_mat, color='red', label='Reconstructed Audio')
    axes[1].set_title('Reconstructed Audio')
    axes[2].plot(recon_mat, color='red', label='Reconstructed Audio (SVR)')
    axes[2].set_title('SVR Reconstructed Audio')
    axes[3].plot(stego_mat, color='green', label='Dropped Audio')
    axes[3].set_title('Dropped Audio')

    for ax in axes:
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.savefig(file_path + 'outputRF.jpg')
    plt.show()


model_name = 'random_forest'
file_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/'

calculations(file_path, model_name)
