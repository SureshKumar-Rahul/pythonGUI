import numpy as np
import pandas as pd
import glob
import math
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras import models, layers, regularizers, optimizers
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from scipy import signal
from joblib import load
from keras.api.models import load_model

def apply_notch_filter(data, fs, freq=50.0):
    Q = 30.0  # Quality factor
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data)

# Bandpass Filter
def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Define FIR Filter Size
def firwdsize(ftype, fl, fh, fs):
    df = (fh - fl) / fs
    if ftype == 1:
        s = 3.1 / df
    elif ftype == 2:
        s = 3.3 / df
    elif ftype == 3:
        s = 5.5 / df
    else:
        s = 0.9 / df
    s = math.ceil(s)
    return s + 1 if s % 2 == 0 else s

# Define FIR windowed filter
def firwd(filtertype, win, fl, fh, fs):
    size = firwdsize(filtertype, fl, fh, fs)
    M = int((size - 1) / 2)
    o1 = (2.0 * np.pi * fl) / fs
    o2 = (2.0 * np.pi * fh) / fs
    h = np.zeros(M + 1)
    w = np.ones(M + 1)
    hw = np.zeros(size)

    for n in range(M, 0, -1):
        if filtertype == 0:
            h[M - n] = np.sin(o1 * n) / (n * np.pi)
        elif filtertype == 1:
            h[M - n] = -np.sin(o1 * n) / (n * np.pi)
        elif filtertype == 2:
            h[M - n] = (np.sin(o2 * n) - np.sin(o1 * n)) / (n * np.pi)
        elif filtertype == 3:
            h[M - n] = (-np.sin(o2 * n) + np.sin(o1 * n)) / (n * np.pi)

    for n in range(M, -1, -1):
        if win == 1:
            w[M - n] = 0.5 + 0.5 * np.cos((n * np.pi) / M)
        elif win == 2:
            w[M - n] = 0.54 + 0.46 * np.cos((n * np.pi) / M)
        elif win == 3:
            w[M - n] = 0.42 + 0.5 * np.cos((n * np.pi) / M) + 0.08 * np.cos((2 * n * np.pi) / M)

    h[M] = {
        0: o1 / np.pi,
        1: (np.pi - o1) / np.pi,
        2: (o2 - o1) / np.pi,
        3: (np.pi - o2 + o1) / np.pi
    }[filtertype]

    hw[:M + 1] = np.multiply(h, w)
    hw[M + 1:] = hw[M - 1::-1]
    return hw

# Function to apply FIR filtering
def apply_fir_filter(data, filtertype, win, fl, fh, fs):
    fir_coeffs = firwd(filtertype, win, fl, fh, fs)
    return signal.lfilter(fir_coeffs, 1.0, data)

# Signal Preprocessing
def preprocess_signal(data, fs):
    # Apply notch filter to remove powerline noise
    data = apply_notch_filter(data, fs)

    # Apply baseline correction (mean subtraction)
    data = data - np.mean(data)

    return data

# Feature extraction function
def extract_features(window):
    features = [
        np.mean(window),
        np.std(window),
        np.mean(np.abs(window - np.mean(window))),
        np.percentile(window, 75) - np.percentile(window, 25),
        np.percentile(window, 75),
        kurtosis(window) if np.var(window) > 1e-10 else 0,
        np.max(window) - np.min(window),
        np.sum(np.power(np.abs(np.fft.fft(window)), 2))
    ]

    # Frequency domain features
    fft_freqs = np.fft.fftfreq(len(window), d=1 / 250)
    fft_vals = np.abs(np.fft.fft(window))
    sum_fft_vals = np.sum(fft_vals)
    features.append(np.sum(np.abs(fft_freqs * fft_vals)) / sum_fft_vals if sum_fft_vals != 0 else 0)

    # Spectral Entropy
    prob_fft = np.abs(fft_vals) ** 2
    prob_fft /= np.sum(prob_fft)
    spectral_entropy = -np.sum(prob_fft * np.log(prob_fft + 1e-10))
    features.append(spectral_entropy)

    # Peak Frequency
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    features.append(peak_freq)

    # AutoReg coefficients
    try:
        model = AutoReg(window, lags=2, old_names=False)
        model_fit = model.fit()
        features.extend(model_fit.params[1:3])
    except Exception:
        features.extend([0, 0])

    return features

# Parameters
window_size = 500  # for 250 samples 2 seconds
fs = 250

# Define the feature names
bands = ['Raw', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
base_feature_names = [
    'Mean', 'Standard Deviation', 'Mean Absolute Deviation',
    'Interquartile Range', '75th Percentile', 'Kurtosis',
    'Range', 'Sum of Squares of FFT', 'Spectral Centroid',
    'Spectral Entropy', 'Peak Frequency',
    'AR Coefficient 1', 'AR Coefficient 2'
]

# Create expanded feature names for each band including Raw data
feature_names = [f'{band}_{feat}' for band in bands for feat in base_feature_names]

# Load the new test data
new_test_files = glob.glob('Data/Music/Real_world_test/*.csv')
new_test_features = []
new_test_labels = []

for file in new_test_files:
    recording = pd.read_csv(file)
    music_track = recording['Track'].unique()[0]
    data = recording.drop(columns=['Track']).to_numpy().T

    for channel_index, channel_data in enumerate(data):
        # Preprocess the raw channel data
        channel_data = preprocess_signal(channel_data, fs)

        raw_features = extract_features(channel_data)

        # Filter the data into frequency bands
        delta_band = apply_fir_filter(channel_data, 2, 2, 0.5, 4, fs)
        theta_band = apply_fir_filter(channel_data, 2, 2, 4, 8, fs)
        alpha_band = apply_fir_filter(channel_data, 2, 2, 8, 13, fs)
        beta_band = apply_fir_filter(channel_data, 2, 2, 13, 30, fs)
        gamma_band = apply_fir_filter(channel_data, 2, 2, 30, 0.49 * fs, fs)

        for start in range(0, len(channel_data), window_size):
            window = channel_data[start:start + window_size]
            if len(window) == window_size:
                delta_features = extract_features(delta_band[start:start + window_size])
                theta_features = extract_features(theta_band[start:start + window_size])
                alpha_features = extract_features(alpha_band[start:start + window_size])
                beta_features = extract_features(beta_band[start:start + window_size])
                gamma_features = extract_features(gamma_band[start:start + window_size])

                features = np.concatenate([
                    raw_features,
                    delta_features,
                    theta_features,
                    alpha_features,
                    beta_features,
                    gamma_features
                ])
                new_test_features.append(features)
                new_test_labels.append(music_track)  # Assuming "Track" is the label

# Convert the features to a DataFrame
new_test_features_df = pd.DataFrame(new_test_features, columns=feature_names)

# Load the saved scaler
scaler = load('scaler_Subject 1.joblib')
model = load_model('best_ann_model_Subject 1.keras')

# Normalize the new test features
X_new_test = scaler.transform(new_test_features_df)

# Convert labels to integer type (assuming they need to be converted)
y_new_test = np.array(new_test_labels).astype(int)

# Predict using the loaded model
y_new_pred = model.predict(X_new_test)
y_new_pred_classes = np.argmax(y_new_pred, axis=1)

# Evaluate the model using the true labels
test_accuracy = accuracy_score(y_new_test, y_new_pred_classes)
print(f"New Test Accuracy: {test_accuracy:.4f}")

# Print classification report
print("Classification Report:\n", classification_report(y_new_test, y_new_pred_classes))

# Optionally save predictions to a file
predictions_df = pd.DataFrame({'True Label': y_new_test, 'Predicted Class': y_new_pred_classes})
predictions_df.to_csv('predictions_real_world_test.csv', index=False)
