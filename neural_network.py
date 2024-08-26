import numpy as np
import pandas as pd
import glob
import math
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras import models, layers, regularizers, optimizers
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from scipy import signal  # Importing the signal module
from joblib import dump

def load_eeg_data(subjects, tracks):
    data_dict = {}
    for subject in subjects:
        for track in tracks:
            file_pattern = f'Data/Music/{subject}/{track}/data_*.csv'
            files = sorted(glob.glob(file_pattern))  # Sort files to maintain order
            data_dict[(subject, track)] = files
    return data_dict

# Parameters
window_size = 500  # for 250 samples 2 seconds
fs = 250

# Load the EEG data
subjects = ['Subject 0']  # , 'Subject 1', 'Subject 2', 'Subject 3']
tracks = ['track0.mp3', 'track1.mp3', 'track2.mp3', 'track3.mp3', 'track4.mp3', 'track5.mp3',
          'track6.mp3', 'track7.mp3', 'track8.mp3', 'track9.mp3']

data_dict = load_eeg_data(subjects, tracks)

# Process data
all_features = []

# Define Notch Filter for power line noise (50/60Hz)
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

# Function to apply FIR filtering
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

    # Optionally downsample if your sampling rate is very high
    # data = signal.resample(data, int(len(data) * target_fs / fs))

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

# Data Augmentation Function
def augment_signal(data):
    augmented_data = []

    # Additive Gaussian Noise
    noise = np.random.normal(0, 0.01, len(data))
    augmented_data.append(data + noise)

    # Time Shifting
    shift = np.random.randint(1, len(data) // 10)
    augmented_data.append(np.roll(data, shift))

    # Scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented_data.append(data * scale)

    # Flipping (inversion)
    augmented_data.append(-data)

    # Time Stretching (using resample, not directly in time)
    stretched_data = signal.resample(data, int(len(data) * 1.1))
    augmented_data.append(stretched_data[:len(data)])  # Ensure same length

    return augmented_data

# Data Preprocessing Loop with Augmentation
for (sub, track), files in data_dict.items():
    for file in files:
        recording = pd.read_csv(file)
        music_track = recording['Track'].unique()[0]
        data = recording.drop(columns=['Track']).to_numpy().T

        for channel_index, channel_data in enumerate(data):
            # Preprocess raw channel data
            channel_data = preprocess_signal(channel_data, fs)

            # Apply augmentation
            augmented_signals = augment_signal(channel_data)

            for aug_signal in [channel_data] + augmented_signals:  # Rename `signal` to `aug_signal`
                # Extract features from the raw (unfiltered) data
                raw_features = extract_features(aug_signal)

                # Filter the data into frequency bands
                delta_band = apply_fir_filter(aug_signal, 2, 2, 0.5, 4, fs)
                theta_band = apply_fir_filter(aug_signal, 2, 2, 4, 8, fs)
                alpha_band = apply_fir_filter(aug_signal, 2, 2, 8, 13, fs)
                beta_band = apply_fir_filter(aug_signal, 2, 2, 13, 30, fs)
                gamma_band = apply_fir_filter(aug_signal, 2, 2, 30, 0.49 * fs, fs)

                for start in range(0, len(aug_signal), window_size):
                    window = aug_signal[start:start + window_size]
                    if len(window) == window_size:
                        # Extract features from each band within the window
                        delta_features = extract_features(delta_band[start:start + window_size])
                        theta_features = extract_features(theta_band[start:start + window_size])
                        alpha_features = extract_features(alpha_band[start:start + window_size])
                        beta_features = extract_features(beta_band[start:start + window_size])
                        gamma_features = extract_features(gamma_band[start:start + window_size])

                        # Concatenate features from the raw data and all frequency bands
                        features = np.concatenate([
                            raw_features,  # Features from the raw data
                            delta_features,
                            theta_features,
                            alpha_features,
                            beta_features,
                            gamma_features
                        ])

                        # Store the feature vector along with metadata
                        feature_dict = {
                            'track': music_track,
                            'channel': channel_index,
                            'features': features
                        }
                        all_features.append(feature_dict)

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

# Convert list of feature dictionaries to DataFrame
all_features_df = pd.DataFrame(all_features)
all_features_df.to_csv("features.csv")

# Add labels (assuming each track corresponds to a label, modify as needed)
all_features_df['label'] = all_features_df['track'].astype(int)

# Shuffle and split the data into training, validation, and test sets
all_features_df = all_features_df.sample(frac=1, random_state=42).reset_index(drop=True)

# First, split the data into training + validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    pd.DataFrame(all_features_df['features'].tolist(), columns=feature_names),
    all_features_df['label'].values,
    test_size=0.1,  # Reserve 10% for the test set
    random_state=42
)

# Now, split the training + validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15,  # 15% of the remaining data for validation
    random_state=42
)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the scaler to a file
dump(scaler, f'scaler_{subjects[0]}.joblib')

# Determine the number of classes
num_classes = len(np.unique(y_train))

# Build and train the ANN model
def build_ann_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the ANN model
model = build_ann_model((X_train.shape[1],), num_classes)

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
mc = ModelCheckpoint(f'best_ann_model_{subjects[0]}.keras', monitor='val_accuracy', mode='max', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    epochs=1000,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Use validation data
                    callbacks=[es, mc, reduce_lr],
                    verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Print classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred_classes))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
