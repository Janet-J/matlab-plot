import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.io import loadmat
from sklearn.svm import LinearSVR
from scipy.io.wavfile import read, write
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
import librosa
import os

## MACHINE LEARNING FOR AUDIO RECONSTRUCTION

def networks(md, X_train, y_train):
    if md != 'random_forest':
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if md == 'lstm':
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=128)

    elif md == 'simple_rnn':
        model = Sequential()
        model.add(SimpleRNN(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(Dense(units=1))

        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=0)

    elif md == 'gru':
        model = Sequential()
        model.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.6, nesterov=False), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)

    elif md == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

    elif md == 'svr':
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples, nx*ny))
        y_train_2d = np.array(y_train)
        y_train_1d = y_train_2d.flatten()

        model = LinearSVR(verbose=0, C=1, epsilon=0, fit_intercept=True, intercept_scaling=1.0, max_iter=2000, random_state=None, tol=0.0001)
        model.fit(X_train_2d, y_train_1d)

    else:
        print('No Such Model')
        return None

    return model

def calculations(file_path, model_name):
    original_mat = pd.DataFrame(loadmat(file_path + 'original.mat')['y'])
    hidden_mat = pd.DataFrame(loadmat(file_path + 'hiddenData.mat')['KK'])
    stego_mat = pd.DataFrame(loadmat(file_path + 'stegoAudio.mat')['W'])
    stego_recon = pd.DataFrame(loadmat(file_path + 'reconstructedAudio.mat')['recon'])

    nan_index = stego_mat[0][stego_mat[0].apply(np.isnan)].index
    stego_mat.iloc[nan_index] = 0

    original_drop_mat = original_mat.copy()
    original_drop_mat.iloc[nan_index] = 0

    hidden_drop_mat = hidden_mat.copy()
    hidden_drop_mat.iloc[nan_index] = 0

    nan_index = nan_index[nan_index < len(stego_mat)]  # Limit the size of nan_index

    test_data = hidden_mat.iloc[nan_index]

    X_train = hidden_drop_mat
    y_train = stego_mat

    model = networks(model_name, X_train, y_train)
    if model is None:
        return

    X_test = np.array(test_data)

    if model_name in ['lstm', 'simple_rnn', 'gru']:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_recon_audio = model.predict(X_test)

    final_mat = stego_mat.copy()
    min_length = min(len(nan_index), len(predicted_recon_audio))

    for i in range(min_length):
        final_mat.iloc[nan_index[i]] = predicted_recon_audio[i]

    signalOut = final_mat.values
    reconstructed_audio = {"reconL": final_mat.values, "Fs": 44100}
    sp.savemat(file_path + model_name + "_reconstructed_audio.mat", reconstructed_audio)
    write(file_path + 'OutAudio/reconstructedsimple_rnn.wav', 44100, signalOut.astype(np.float64))

    recon_mat = pd.DataFrame(final_mat.values)
    frames = pd.concat([original_mat, recon_mat, stego_mat, stego_recon], axis=1)
    frames.columns = ['original', model_name, 'stego_audio', 'stego_recon']
    print(frames.corr())

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    axes[0].plot(original_mat, color='blue', label='Original Audio')
    axes[0].set_title('Original Audio')
    axes[1].plot(recon_mat, color='red', label='Reconstructed Audio (simple_rnn)')
    axes[1].set_title('simple_rnn Reconstructed Audio')
    axes[2].plot(stego_mat, color='green', label='Dropped Audio')
    axes[2].set_title('Dropped Audio')

    for ax in axes:
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\simple_rnn_plots\outputsimple_rnn.jpg')
    plt.show()

# Run the model
model_name = 'simple_rnn'
file_path =  'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/'
#calculations(file_path, model_name)
print("Done")

  #Spectral Centroid
folder_path = r'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'
    
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Set up the subplot grid size
num_files = len(wav_files)
cols = 2  # Number of columns in subplot
rows = (num_files // cols) + (num_files % cols > 0)  # Calculate required rows

# Create a figure for subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))

# If only one subplot, make `axes` a list
if num_files == 1:
    axes = [axes]
else:
    axes = axes.flatten()  # Flatten the array for easier indexing

# Dictionary to store spectral centroids and bandwidths
spectral_centroids = {}
spectral_bandwidths = {}

for i, wav_file in enumerate(wav_files):
    file_path = os.path.join(folder_path, wav_file)
    
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Compute Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=512)
    mean_centroid = np.mean(centroid)  # Convert to mean spectral centroid
    
    # Compute Spectral Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024, hop_length=512))
    
    # Store in dictionaries
    spectral_centroids[wav_file] = mean_centroid
    spectral_bandwidths[wav_file] = bandwidth
    
    # Plot the waveform and spectral centroid
    times = librosa.times_like(centroid, sr=sr, hop_length=512)
    
    #  Fix: Ensure `times` and `centroid[0]` have the same shape
    axes[i].plot(times[:centroid.shape[1]], centroid[0], label="Spectral Centroid", color='r')
    
    axes[i].set_title(f"{wav_file}\nCentroid: {mean_centroid:.2f} Hz\nBandwidth: {bandwidth:.2f} Hz")
    axes[i].legend()
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Frequency (Hz)")

# Adjust layout
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\centroid_plots\outputsignal1.jpg')
plt.show()

# Print results
print("\nSpectral Centroid and Bandwidth Results:")
for file in wav_files:
    print(f"{file}: Centroid = {spectral_centroids[file]:.2f} Hz, Bandwidth = {spectral_bandwidths[file]:.2f} Hz")



#MSE Plots

# Define the folder path where .wav files are stored
folder_path = r'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Identify the original and reconstructed files
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Assuming the original file has "original" in its name
original_file = [f for f in wav_files if "original" in f.lower()]
if not original_file:
    raise FileNotFoundError("No original file found in folder A.")
original_file = original_file[0]  # Get the first match

# Get reconstructed files (exclude the original)
reconstructed_files = [f for f in wav_files if f != original_file]

# Load the original audio file
original_audio, sr = librosa.load(os.path.join(folder_path, original_file))

# Dictionary to store MSE values
mse_values = {}

for rec_file in reconstructed_files:
    # Load the reconstructed audio file
    reconstructed_audio, sr_rec = librosa.load(os.path.join(folder_path, rec_file))
    
    # Ensure both signals have the same length
    min_length = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_length]
    reconstructed_trimmed = reconstructed_audio[:min_length]
    
    # Compute MSE
    mse = np.mean((original_trimmed - reconstructed_trimmed) ** 2)
    mse_values[rec_file] = mse

# Plot MSE values
plt.figure(figsize=(8, 5))
plt.bar(mse_values.keys(), mse_values.values(), color='skyblue')
plt.xlabel("Reconstructed Files")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE between Original and Reconstructed Audio Files")
plt.xticks(rotation=15)  # Rotate x labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\MSE_plots\outputsignal1.jpg')
plt.show()

# Print MSE values for reference
print("\nMSE Values:")
for file, mse in mse_values.items():
    print(f"{file}: {mse:.6f}")



# Correlation
# Define the folder path where .wav files are stored
folder_path = r'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Identify the original and reconstructed files
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Assuming the original file has "original" in its name
original_file = [f for f in wav_files if "original" in f.lower()]
if not original_file:
    raise FileNotFoundError("No original file found in folder A.")
original_file = original_file[0]  # Get the first match

# Get reconstructed files (exclude the original)
reconstructed_files = [f for f in wav_files if f != original_file]

# Load the original audio file
original_audio, sr = librosa.load(os.path.join(folder_path, original_file))

# Dictionary to store correlation values
correlation_values = {}

for rec_file in reconstructed_files:
    # Load the reconstructed audio file
    reconstructed_audio, sr_rec = librosa.load(os.path.join(folder_path, rec_file))
    
    # Ensure both signals have the same length
    min_length = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_length]
    reconstructed_trimmed = reconstructed_audio[:min_length]
    
    # Compute correlation coefficient
    correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
    correlation_values[rec_file] = correlation

# Plot Correlation values
plt.figure(figsize=(8, 5))
plt.bar(correlation_values.keys(), correlation_values.values(), color='lightgreen')
plt.xlabel("Reconstructed Files")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation between Original and Reconstructed Audio Files")
plt.xticks(rotation=15)  # Rotate x labels for readability
plt.ylim(0, 1)  # Correlation ranges from -1 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
if correlation_values:
    print("\nCorrelation values computed successfully! Plotting now...")
    plt.show()
else:
    print("\nError: No correlation values found. Check if files were correctly loaded.")

plt.show()

# Print Correlation values for reference
print("\nCorrelation Values:")
for file, corr in correlation_values.items():
    print(f"{file}: {corr:.6f}")
