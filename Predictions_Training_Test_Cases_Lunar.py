import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from obspy import read
import pickle
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

cat_directory = './data/lunar/training/catalogs/'
data_directory = './data/lunar/training/data/S12_GradeA/'
output_directory = './Resources/lunar/Results'
resources_directory = './Resources/lunar'
def load_catalog(cat_file):
    try:
        return pd.read_csv(cat_file)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return None

def load_seismic_data(filename):
    try:
        mseed_file = os.path.join(data_directory, f"{filename}.mseed")
        return read(mseed_file)
    except Exception as e:
        print(f"Error loading seismic data: {e}")
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


def plot_comparison(tr_times, tr_data, arrival, predicted_time, test_case):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tr_times, tr_data, label='Seismic Data', color='black')
    ax.axvline(x=arrival, color='red', linestyle='--', label='Original Arrival Time')
    ax.axvline(x=predicted_time, color='blue', linestyle='--', label='Predicted Start Time')
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'{test_case} - Arrival and Predicted Start Times', fontweight='bold')
    ax.legend()

    output_path = os.path.join(output_directory, f'{test_case}_comparison_plot.png')
    plt.savefig(output_path)
    plt.close()

def predict_and_compare(catalog, model_filename, scaler_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    predictions_df = pd.read_csv(os.path.join(resources_directory, 'test_predictions.csv'))

    for index, row in predictions_df.iterrows():
        test_filename = row['Filename']  
        predicted_time = row['Predicted']  

        st = load_seismic_data(test_filename)

        if st is not None and len(st.traces) > 0:
            trace = st.traces[0] 
            
            velocity_features = extract_velocity_features(trace)
            feature_dict = velocity_features
            X = pd.DataFrame([feature_dict])  
            X_scaled = scaler.transform(X) 

            tr_times = np.arange(trace.stats.npts) / trace.stats.sampling_rate
            tr_data = np.array(trace.data)

            test_case_name = f'Test_Case_{index + 1}'
            plot_comparison(tr_times, tr_data, row['Actual'], predicted_time, test_case_name)
        else:
            print(f"Warning: No seismic data found or no traces for {test_filename}")

if __name__ == "__main__":
    catalog_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')
    catalog = load_catalog(catalog_file)
    
    if catalog is not None:
        model_filename = os.path.join(resources_directory, "best_random_forest_model.pkl")
        scaler_filename = os.path.join(resources_directory, "scaler.pkl")
        predict_and_compare(catalog, model_filename, scaler_filename)
