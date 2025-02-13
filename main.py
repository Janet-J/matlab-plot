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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
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
        model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=[early_stop], verbose=1)

    elif md == 'simple_rnn':
        model = Sequential()
        model.add(SimpleRNN(256, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(256, return_sequences=True))
        model.add(SimpleRNN(256, return_sequences=True))
        model.add(Dense(units=1))

        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)

    elif md == 'gru':
        model = Sequential()
        model.add(GRU(units=265, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=128, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=64, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.6, nesterov=False), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=1)

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
    reconstructed_audio = {"reconL": final_mat.values, "Fs": 44100}
    sp.savemat(file_path + model_name + "_reconstructed_audio.mat", reconstructed_audio)
    write(file_path + 'OutAudio/reconstructedrandom_forest1.wav', 44100, signalOut.astype(np.float64))

    recon_mat = pd.DataFrame(final_mat.values)
    frames = pd.concat([original_mat, recon_mat, stego_mat, stego_recon], axis=1)
    frames.columns = ['original', model_name, 'stego_audio', 'stego_recon']
    print(frames.corr())

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    axes[0].plot(original_mat, color='blue', label='Original Audio')
    axes[0].set_title('Original Audio')
    axes[1].plot(stego_recon, color='red', label='Reconstructed Audio (random_forest)')
    axes[1].set_title('random_forest Reconstructed Audio')
    axes[2].plot(stego_mat, color='green', label='Dropped Audio')
    axes[2].set_title('Dropped Audio')

    for ax in axes:
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\simple_rnn_plots\outputrandom_forest1.jpg')
    plt.show()

# Run the model
model_name = 'random_forest'
file_path =  'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/'
#calculations(file_path, model_name)
print("Done")




# Folder path where your .wav files are stored
folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Get all .wav files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Initialize the plot
plt.figure(figsize=(12, 6))

# Iterate through all the .wav files and plot them
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    
    # Load the .wav file
    sample_rate, audio_data = wavfile.read(file_path)

    # Normalize the audio data to ensure all signals fit in the same range
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)  # Avoid division by zero

    # Plot the waveform
    plt.plot(audio_data, label=f'{wav_file}')

# Add title, labels, legend, and grid
plt.title('Comparison of Waveforms from OutAudio Folder')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Amplitude')
plt.legend(loc='upper right')
plt.grid()

# Display the plot
plt.tight_layout()
plt.show()

#*******************************************************************************************
#plotting the  sapmle sample_rate
#***************************************************************************************
folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Get all .wav files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
# Create a new figure for waveform plots


# Plot the waveform of each audio file
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    sample_rate, audio_data = wavfile.read(file_path)

    # Normalize the audio to avoid plotting errors
    audio_data = audio_data / np.max(np.abs(audio_data))  # Optional normalization

    plt.plot(audio_data, label=f'{wav_file} (Sample Rate: {sample_rate})')

plt.title('Waveform of Multiple .wav Files')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.savefig('centroid_plots/waveform_output.jpg')
plt.show()



# Create a figure for the plot
plt.figure(figsize=(12, 6))
spectral_centroids= []

# Iterate through each audio file and calculate FFT
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    sample_rate, audio_data = wavfile.read(file_path)

    # Normalize amplitude (if needed)
    audio_data = audio_data.astype(float)  # Ensure correct dtype
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)  # Avoid division by zero

    # FFT calculation
    fft_result = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft_result)[:len(fft_result) // 2]  # Keep only positive frequencies

    freq = np.fft.fftfreq(len(audio_data), d=1/sample_rate)[:len(fft_result) // 2]

    # Avoid zero or near-zero magnitude affecting centroid calculation
    fft_magnitude += 1e-10

    # Calculate the spectral centroid
    spectral_centroid = np.sum(freq * fft_magnitude) / np.sum(fft_magnitude)
    spectral_centroids.append((wav_file, spectral_centroid))

    # Plot frequency spectrum
    plt.plot(freq, 20 * np.log10(fft_magnitude), label=f"{wav_file} (centroid: {spectral_centroid:.2f} Hz)")

    plt.xscale("log")
    print(f'processed:{wav_file}, spectral Centroid: {spectral_centroid} Hz')

plt.title('Frequency Spectrum of Multiple .wav Files')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.savefig('centroid_plots/output1')
plt.show()

print('\nSpectral Centroids for all Files:')
for wav_file, centroid in spectral_centroids:
    print(f'{wav_file}: {centroid:.2f} Hz')



#Calculating the MSE and correlation

folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Load original and reconstructed files
# Load original and reconstructed files
original_file = os.path.join(folder_path, "original.wav")
sample_rate_original, original_audio = wavfile.read(original_file)

# Ensure normalization for the original signal (avoid division by zero)
original_audio = original_audio / (np.max(np.abs(original_audio)) + 1e-10)

# Get all reconstructed .wav files in the folder
reconstructed_files = [f for f in os.listdir(folder_path) if f.startswith("reconstructed") and f.endswith(".wav")]

# Iterate through reconstructed files for comparison
for recon_file in reconstructed_files:
    recon_path = os.path.join(folder_path, recon_file)
    sample_rate_recon, recon_audio = wavfile.read(recon_path)

    # Normalize reconstructed signal
    recon_audio = recon_audio / (np.max(np.abs(recon_audio)) + 1e-10)

    # Resample original audio to match reconstructed audio length
    original_resampled = resample(original_audio, len(recon_audio))

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_resampled - recon_audio) ** 2)

    # Calculate correlation coefficient
    correlation = np.corrcoef(original_resampled, recon_audio)[0, 1]

    # Print results
    print(f"\nComparison with {recon_file}:")
    print(f"Original Signal Mean: {np.mean(original_resampled):.4f}, Std: {np.std(original_resampled):.4f}")
    print(f"Reconstructed Signal Mean: {np.mean(recon_audio):.4f}, Std: {np.std(recon_audio):.4f}")
    print(f"MSE: {mse:.4f}, Correlation: {correlation:.4f}")

# Plot original and reconstructed signals
    plt.figure(figsize=(12, 6))
plt.plot(original_resampled, label="Original Signal", alpha=0.7)
plt.plot(recon_audio, label=f"Reconstructed Signal ({recon_file})", alpha=0.7)
plt.title(f"Original vs {recon_file}\nMSE: {mse:.2f}, Correlation: {correlation:.2f}")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()


folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

# Load original audio
original_file = os.path.join(folder_path, "original.wav")
sample_rate_original, original_audio = wavfile.read(original_file)

# Normalize the original audio
original_audio = original_audio / (np.max(np.abs(original_audio)) + 1e-10)

# Get all reconstructed .wav files in the folder
reconstructed_files = [f for f in os.listdir(folder_path) if f.startswith("reconstructed") and f.endswith(".wav")]

# Lists to store MSE and correlation values
mse_values = []
correlation_values = []
file_names = []

# Iterate through each reconstructed file
for recon_file in reconstructed_files:
    recon_path = os.path.join(folder_path, recon_file)
    sample_rate_recon, recon_audio = wavfile.read(recon_path)

    # Normalize the reconstructed audio
    recon_audio = recon_audio / (np.max(np.abs(recon_audio)) + 1e-10)

    # Resample original audio to match the length of the reconstructed audio
    original_resampled = resample(original_audio, len(recon_audio))

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_resampled - recon_audio) ** 2)

    # Calculate Correlation
    correlation = np.corrcoef(original_resampled, recon_audio)[0, 1]

    # Store values for plotting
    mse_values.append(mse)
    correlation_values.append(correlation)
    file_names.append(recon_file)

    # Print diagnostic information
    print(f"\nComparison with {recon_file}:")
    print(f"MSE: {mse:.4f}, Correlation: {correlation:.4f}")

# Plot all MSE values on one graph
plt.figure(figsize=(10, 6))
plt.bar(file_names, mse_values, color='orange')
plt.title('MSE for All Reconstructed Audio Files')
plt.xlabel('Reconstructed File')
plt.ylabel('MSE Value')
plt.xticks(rotation=45, ha="right")  # Rotate file names for better visibility
plt.tight_layout()
plt.grid()
plt.savefig('MSE_plots/output1')
plt.show()




# Set the folder path containing the .wav files
folder_path = 'C:/Users/HP/Desktop/TestProject/audio_reconstruction_project/OutAudio'

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
plt.savefig(r'C:\Users\HP\Desktop\TestProject\audio_reconstruction_project\OutAudioplots\output1')
# Display the plot
plt.show()


