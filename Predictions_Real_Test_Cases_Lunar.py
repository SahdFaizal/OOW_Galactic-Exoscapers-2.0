import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from obspy import read
import pickle
from scipy.signal import find_peaks

# Define directories
data_directories = [
    './data/lunar/test/data/S16_GradeB/',
    './data/lunar/test/data/S16_GradeA/',
    './data/lunar/test/data/S15_GradeB/',
    './data/lunar/test/data/S15_GradeA/',
    './data/lunar/test/data/S12_GradeB/'
]
output_directory = './Resources/lunar/Results'
resources_directory = './Resources/lunar'
output_catalog_directory = '.'

# Ensure output catalog directory exists
os.makedirs(output_catalog_directory, exist_ok=True)

def load_seismic_data(mseed_file):
    """Load seismic data from an mseed file."""
    try:
        return read(mseed_file)
    except Exception as e:
        print(f"Error loading seismic data from {mseed_file}: {e}")
    return None
def extract_velocity_features(tr):
    data = tr.data
    fs = tr.stats.sampling_rate
    features = {}

    if len(data) == 0:
        return {f'feature_{i}': np.nan for i in range(10)}


    try:
        peaks, properties = find_peaks(data, height=0)
        peak_heights = properties['peak_heights']

        # Temporal Features
        features['Mean_Signal_Level'] = np.mean(data)
        features['Zero_Crossing_Rate'] = ((data[:-1] * data[1:]) < 0).sum() / len(data)

        time_array = np.arange(len(data)) / fs

        if len(peaks) > 0:
            N = min(500, len(peaks))
            top_peaks_indices = np.argsort(peak_heights)[-N:][::-1]
            selected_peak_heights = peak_heights[top_peaks_indices]

            # Calculate displacement by integrating velocity
            displacement = np.cumsum(data) / fs
            
            # Peak spatial displacement
            features['Peak_Spatial_Displacement_Time'] = time_array[np.argmax(displacement)]
            features['Mean_Spatial_Displacement_Time'] = time_array[np.argmax(displacement)]
            features['Spatial_Variance'] = np.var(displacement)

            selected_peak_times = time_array[peaks[top_peaks_indices]]

            # Distance Between Peaks
            if len(selected_peak_times) > 1:
                peak_intervals = np.diff(selected_peak_times)
                features['Distance_Between_Peaks'] = np.mean(peak_intervals * np.mean(np.diff(displacement[peaks[top_peaks_indices]])) / np.mean(np.diff(selected_peak_times)))
            else:
                features['Distance_Between_Peaks'] = np.nan

            # Other Peak Features
            features['Top_Peaks_Mean'] = np.mean(selected_peak_heights)
            features['Top_Peaks_Mean_Time'] = selected_peak_times[np.argmax(selected_peak_heights)]
            features['Top_Peaks_Std_Time'] = selected_peak_times[np.argmax(selected_peak_heights)]
            features['Top_Peaks_Median'] = np.median(selected_peak_heights)
            features['Top_Peaks_Median_Time'] = selected_peak_times[np.argmin(np.abs(selected_peak_heights - np.median(selected_peak_heights)))]

            features['Top_Peaks_Max_Time'] = selected_peak_times[np.argmax(selected_peak_heights)]
            features['Top_Peaks_Min_Time'] = selected_peak_times[np.argmin(selected_peak_heights)]

            # First and second derivatives
            if len(selected_peak_heights) > 1:
                first_derivative = np.diff(selected_peak_heights)
                features['First_Derivative_Std_Time'] = selected_peak_times[np.argmax(first_derivative)]

                second_derivative = np.diff(first_derivative)
                features['Second_Derivative_Mean'] = np.mean(second_derivative)
                features['Second_Derivative_Mean_Time'] = selected_peak_times[np.argmax(second_derivative)]

            # Spectral Entropy
            power_spectrum = np.abs(np.fft.fft(data))**2
            normalized_power = power_spectrum / np.sum(power_spectrum)
            features['Spectral_Entropy'] = -np.sum(normalized_power * np.log(normalized_power + 1e-12))

        else:
            features = {f'feature_{i}': np.nan for i in range(10)}

    except Exception as e:
        print(f"Error extracting features: {e}")
        features = {f'feature_{i}': np.nan for i in range(10)}

    return features


def predict_and_compare(model_filename, scaler_filename):
    """Predict seismic events based on extracted features from .mseed files."""
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    predictions = []

    # Iterate through each data directory and process .mseed files
    for directory in data_directories:
        for filename in os.listdir(directory):
            if filename.endswith('.mseed'):
                mseed_file = os.path.join(directory, filename)
                st = load_seismic_data(mseed_file)

                if st is not None and len(st.traces) > 0:
                    trace = st.traces[0]
                    velocity_features = extract_velocity_features(trace)
                    feature_dict = velocity_features
                    X = pd.DataFrame([feature_dict])
                    X_scaled = scaler.transform(X) 

                    # Predict the time based on features
                    predicted_time = model.predict(X_scaled)[0]  # Assuming model returns predicted time
                    predictions.append({
                        'Filename': filename,
                        'Predicted_Time': predicted_time
                    })
                else:
                    print(f"Warning: No seismic data found for {filename}")

    # Save predictions to a catalog file
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_catalog_directory, 'Output_Catalog_Lunar.csv'), index=False)

if __name__ == "__main__":
    model_filename = os.path.join(resources_directory, "best_random_forest_model.pkl")
    scaler_filename = os.path.join(resources_directory, "scaler.pkl")
    predict_and_compare(model_filename, scaler_filename)
