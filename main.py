import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.io import loadmat
from sklearn.svm import LinearSVR
from scipy.io import wavfile
from scipy.io.wavfile import read, write
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import resample
from sklearn.metrics import mean_squared_error
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import librosa
import librosa.display
import os

## MACHINE LEARNING FOR AUDIO RECONSTRUCTION

def networks(md, X_train, y_train):
    if md != 'random_forest':
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if md == 'lstm':
        model = Sequential()
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[early_stop], verbose=1)

    elif md == 'simple_rnn':
        model = Sequential()
        model.add(SimpleRNN(128, return_sequences=True, ))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(SimpleRNN(128, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)

    elif md == 'gru':
        model = Sequential()
        model.add(GRU(units=265, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=32))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='Adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=25, batch_size=128, verbose=1)

    elif md == 'random_forest':
        model = RandomForestRegressor(n_estimators=200, random_state=42)
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
    reconstructed_audio = {"reconL": final_mat.values, "Fs": 22050}
    sp.savemat(file_path + model_name + "_reconstructed_audio.mat", reconstructed_audio)
    write(file_path + '/OutAudio5/reconstructed.wav', 22050, signalOut.astype(np.float64))
    #reconstructed_audio = {"reconL": final_mat.values, "Fs": float(22050)}
    stego_audio = stego_mat.values.flatten()

# Save the stego audio as a .wav file
    write(file_path + '/OutAudio5/stego_audio.wav', 22050, stego_audio.astype(np.float64))

    recon_mat = pd.DataFrame(final_mat.values)
    frames = pd.concat([original_mat, recon_mat, stego_mat, stego_recon], axis=1)
    frames.columns = ['original', model_name, 'stego_audio', 'stego_recon']
    print(frames.corr())

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    axes[0].plot(original_mat, color='blue', label='Original Audio')
    axes[0].set_title('Original Audio')
    axes[1].plot(stego_recon, color='red', label='Reconstructed Audio (gru)')
    axes[1].set_title('gru Reconstructed Audio')
    axes[2].plot(stego_mat, color='green', label='Dropped Audio')
    axes[2].set_title('Dropped Audio')

    for ax in axes:
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    #plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\plots\outputlstm.jpg')
    plt.show()

# svr
model_name = 'lstm'
file_path =  './'
#calculations(file_path, model_name)
print("Done")

# Set the folder path containing the .wav files
folder_path = './OutAudio1'

# Get all .wav files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Iterate through each file and print its sample rate
for wav_file in wav_files:
    # Get the full path of the file
    file_path = os.path.join(folder_path, wav_file)
    
    # Read the .wav file to get its sample rate
    sample_rate, _ = wavfile.read(file_path)
    
    # Output the file name and sample rate
    print(f"File: {wav_file}, Sample Rate: {sample_rate} Hz")
    


# calculating Spectral centroid


# Folder path containing the .wav files
folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio1'
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Step 2: Calculate Spectral Centroids and plot the frequency spectrum
plt.figure(figsize=(12, 6))
spectral_centroids = []

# Loop through each .wav file and calculate spectral centroids
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    sample_rate, audio_data = wavfile.read(file_path)

    # Normalize audio data
    audio_data = audio_data.astype(float)
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    # FFT calculation
    fft_result = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft_result)[:len(fft_result)//2]
    freq = np.fft.fftfreq(len(audio_data), d=1/sample_rate)[:len(fft_result)//2]

    # Avoid zero or near-zero magnitude
    fft_magnitude += 1e-10

    # Calculate spectral centroid
    spectral_centroid = np.sum(freq * fft_magnitude) / np.sum(fft_magnitude)
    spectral_centroids.append((os.path.basename(wav_file), spectral_centroid))

    # Plot frequency spectrum
    plt.plot(freq, 20 * np.log10(fft_magnitude), label=f"{os.path.basename(wav_file)} (centroid: {spectral_centroid:.2f} Hz)")

    print(f"Processed: {os.path.basename(wav_file)}, Spectral Centroid: {spectral_centroid:.2f} Hz")

# Step 3: Plot frequency spectrum for all files
plt.title('Frequency Spectrum of Multiple .wav Files')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Create bar chart to compare spectral centroids (from the files processed)
labels = [file[0] for file in spectral_centroids]  # Get the names of the files
centroids = [file[1] for file in spectral_centroids]  # Get the spectral centroid values
colors = plt.cm.get_cmap("tab10", len(spectral_centroids)).colors  # Color map for visual distinction

# Create bar chart for spectral centroids comparison
plt.figure(figsize=(8, 5))
plt.bar(labels, centroids, color=colors)

# Labels and title for the bar chart
plt.ylabel("Spectral Centroid (Hz)")
plt.title("Comparison of Spectral Centroids")

# Rotate x-axis labels for readability
plt.xticks(rotation=30, ha="right")

# Show values on top of bars
for i, v in enumerate(centroids):
    plt.text(i, v + 20, f"{v:.2f}", ha="center", fontsize=10)

# Display the bar chart
plt.show()


#calculate MSE

# Define folder path containing audio files
folder_path = './OutAudio1'

# Load original file
original_file = os.path.join(folder_path, "pop.00013.wav")
sample_rate_original, original_audio = wavfile.read(original_file)

# Convert original audio to float and normalize
original_audio = original_audio.astype(np.float32)
original_audio /= np.max(np.abs(original_audio)) + 1e-10  # Prevent division by zero

# Get all reconstructed .wav files in the folder
reconstructed_files = [f for f in os.listdir(folder_path) if f.startswith("reconstructed") and f.endswith(".wav")]

mse_values = []
correlation_values = []
file_names = []

# Iterate through each reconstructed file and compare with original
for recon_file in reconstructed_files:
    recon_path = os.path.join(folder_path, recon_file)
    sample_rate_recon, recon_audio = wavfile.read(recon_path)

    # Convert reconstructed audio to float and normalize
    recon_audio = recon_audio.astype(np.float32)
    recon_audio /= np.max(np.abs(recon_audio)) + 1e-10  # Prevent division by zero

    # Step 1: Resample to match sample rates
    if sample_rate_original != sample_rate_recon:
        recon_audio = librosa.resample(recon_audio, orig_sr=sample_rate_recon, target_sr=sample_rate_original)

    # Step 2: Trim or pad to match length
    min_length = min(len(original_audio), len(recon_audio))
    original_resampled = original_audio[:min_length]
    recon_audio = recon_audio[:min_length]

    # Step 3: Apply high-pass filtering to remove low-frequency noise
    original_resampled = librosa.effects.preemphasis(original_resampled)
    recon_audio = librosa.effects.preemphasis(recon_audio)

    # Step 4: Dynamic Range Matching
    recon_audio *= np.max(np.abs(original_resampled)) / (np.max(np.abs(recon_audio)) + 1e-10)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_resampled - recon_audio) ** 2)
    mse_values.append(mse)

    # Compute correlation coefficient
    correlation = np.corrcoef(original_resampled, recon_audio)[0, 1]
    correlation_values.append(correlation)
    file_names.append(recon_file)

    # Print comparison metrics
    print(f"\nComparison with {recon_file}:")
    print(f"Original Signal Mean: {np.mean(original_resampled):.4f}, Std: {np.std(original_resampled):.4f}")
    print(f"Reconstructed Signal Mean: {np.mean(recon_audio):.4f}, Std: {np.std(recon_audio):.4f}")
    print(f"MSE: {mse:.6f}, Correlation: {correlation:.6f}")

# Plot MSE values
plt.figure(figsize=(10, 5))
plt.bar(file_names, mse_values, color='b')
plt.xlabel("Reconstructed Files")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE for Each Reconstructed Audio File")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid()
plt.show()

# Plot Correlation values
plt.figure(figsize=(10, 5))
plt.bar(file_names, correlation_values, color='g')
plt.xlabel("Reconstructed Files")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation between Original and Reconstructed Audio Files")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid()
plt.show()

# Set the folder path containing the .wav files
folder_path = './OutAudio1'

# Get all .wav files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Determine grid size: at least 2 rows, adjust cols as needed
num_files = len(wav_files)
num_cols = 2
num_rows = (num_files + 1) // num_cols

# Create a figure with a grid layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Flatten axes for easy iteration (in case of more than 2x2 layout)
axes = axes.flatten()

# Iterate over each file and plot in a subplot
for i, wav_file in enumerate(wav_files):
    # Load audio data
    sample_rate, audio_data = wavfile.read(os.path.join(folder_path, wav_file))

    # Normalize the audio data
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    # Plot in the corresponding subplot
    axes[i].plot(audio_data)
    axes[i].set_title(f"{wav_file}")
    axes[i].set_xlabel("Sample Index")
    axes[i].set_ylabel("Normalized Amplitude")
    axes[i].grid(True)

# Hide any unused subplots if fewer than total grid cells
for j in range(num_files, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()
#plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\OutAudioplots\output4')
# Display the plot
plt.show()



