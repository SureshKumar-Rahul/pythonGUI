import numpy as np
import pandas as pd
import glob
import math
import joblib  # For saving and loading the model
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from scipy import signal

# Import custom functions from emotive_eeg (if required)
from emotive_eeg import draw_freq_response

# Define FIR filter size
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
    filtered_data = signal.lfilter(fir_coeffs, 1.0, data)
    return filtered_data

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
    fft_freqs = np.fft.fftfreq(len(window), d=1 / 250)
    fft_vals = np.abs(np.fft.fft(window))
    sum_fft_vals = np.sum(fft_vals)
    features.append(np.sum(np.abs(fft_freqs * fft_vals)) / sum_fft_vals if sum_fft_vals != 0 else 0)

    try:
        model = AutoReg(window, lags=2, old_names=False)
        model_fit = model.fit()
        features.extend(model_fit.params[1:3])
    except Exception:
        features.extend([0, 0])

    return features

# Define the feature names
feature_names = [
    'Mean', 'Standard Deviation', 'Mean Absolute Deviation',
    'Interquartile Range', '75th Percentile', 'Kurtosis',
    'Range', 'Sum of Squares of FFT', 'Spectral Centroid',
    'AR Coefficient 1', 'AR Coefficient 2'
]

# Load the EEG data
def load_eeg_data(subjects, tracks):
    data_dict = {}
    for subject in subjects:
        for track in tracks:
            file_pattern = f'Data/Music/{subject}/{track}/data_*.csv'
            files = sorted(glob.glob(file_pattern))  # Sort files to maintain order
            data_dict[(subject, track)] = files
    return data_dict

# Parameters
window_size = 500
fs = 250

# Load the EEG data
subjects = ['Subject 0']#, 'Subject 1', 'Subject 2', 'Subject 3']
tracks = ['track0.mp3', 'track1.mp3', 'track2.mp3', 'track3.mp3', 'track4.mp3', 'track5.mp3',
          'track6.mp3', 'track7.mp3', 'track8.mp3', 'track9.mp3']

data_dict = load_eeg_data(subjects, tracks)

# Process data and train model for each subject
subject_models = {}

for subject in subjects:
    all_features = []

    for (sub, track), files in data_dict.items():
        if sub != subject:
            continue  # Only process the current subject

        for file in files:
            recording = pd.read_csv(file)
            data = recording.to_numpy().T

            for channel_index, channel_data in enumerate(data):
                delta_band = apply_fir_filter(channel_data, 2, 2, 0.5, 4, fs)
                theta_band = apply_fir_filter(channel_data, 2, 2, 4, 8, fs)
                alpha_band = apply_fir_filter(channel_data, 2, 2, 8, 13, fs)
                beta_band = apply_fir_filter(channel_data, 2, 2, 13, 30, fs)
                gamma_band = apply_fir_filter(channel_data, 2, 2, 30, 0.49 * fs, fs)

                for start in range(0, len(channel_data), window_size):
                    window = channel_data[start:start + window_size]
                    if len(window) == window_size:
                        features = extract_features(window)
                        feature_dict = {
                            'track': track,
                            'channel': channel_index,
                            'features': features
                        }
                        all_features.append(feature_dict)

    # Convert list of feature dictionaries to DataFrame
    all_features_df = pd.DataFrame(all_features)

    # Add labels (assuming each track corresponds to a label, modify as needed)
    all_features_df['label'] = all_features_df['track'].apply(lambda x: int(x.split('track')[1].split('.mp3')[0]))

    # Shuffle and split the data into training and test sets
    all_features_df = all_features_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(all_features_df) * 0.8)  # 80% for training, 20% for testing
    train_features_df = all_features_df.iloc[:split_index]
    test_features_df = all_features_df.iloc[split_index:]

    # Normalize the features
    scaler = StandardScaler()
    normalized_train_features = scaler.fit_transform(pd.DataFrame(train_features_df['features'].tolist(), columns=feature_names))
    normalized_test_features = scaler.transform(pd.DataFrame(test_features_df['features'].tolist(), columns=feature_names))

    # Prepare feature vectors and labels
    X_train = normalized_train_features
    y_train = train_features_df['label'].values
    X_test = normalized_test_features
    y_test = test_features_df['label'].values

    # Train a Random Forest classifier for this subject
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Save the model and scaler
    model_filename = f"random_forest_subject_{subject}.joblib"
    scaler_filename = f"scaler_subject_{subject}.joblib"
    joblib.dump(clf, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model and scaler saved as {model_filename} and {scaler_filename}")

    # Reload the model and scaler (for testing)
    loaded_clf = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    normalized_test_features = loaded_scaler.transform(
        pd.DataFrame(test_features_df['features'].tolist(), columns=feature_names))
    X_test = normalized_test_features

    # Make predictions using the loaded model
    y_pred = loaded_clf.predict(X_test)

    # Evaluate the model
    print(f"Subject {subject} - Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Subject {subject} - Classification Report:\n", classification_report(y_test, y_pred))

# Example plots for one of the subjects (replace with actual use cases as needed)
draw_freq_response([delta_band, theta_band, alpha_band, beta_band, gamma_band])
