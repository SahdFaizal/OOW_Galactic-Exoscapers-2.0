import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from obspy import read
import pickle
from scipy.signal import find_peaks

# Define directories
project_name = 'mars'  # Change this to 'moon' for the moon project
cat_directory = f'./data/{project_name}/training/catalogs/'
data_directory = f'./data/{project_name}/training/data/'
output_directory = f'./Resources/{project_name}/'

os.makedirs(output_directory, exist_ok=True)

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



def create_feature_dataframe(cat):
    features = []
    for _, row in cat.iterrows():
        test_filename = row['filename']
        st = load_seismic_data(test_filename)
        if st is not None:
            tr = st[0]
            velocity_features = extract_velocity_features(tr)
            feature_dict = velocity_features
            feature_dict['time_rel(sec)'] = row['time_rel(sec)']
            feature_dict['filename'] = test_filename
            features.append(feature_dict)

    return pd.DataFrame(features)

def train_and_predict_model(feature_data):
    # No splitting; use all data for training
    X = feature_data.drop(columns=['time_rel(sec)', 'filename'])
    y = feature_data['time_rel(sec)']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=100)
    gb_model.fit(X_scaled, y)

    # Predictions
    y_pred = gb_model.predict(X_scaled)

    mse = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Mean Absolute Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save predictions to CSV
    results_df = pd.DataFrame({'Filename': feature_data['filename'], 'Actual': y, 'Predicted': y_pred})
    results_df.to_csv(os.path.join(output_directory, f'{project_name}_predictions.csv'), index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.7, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, f'{project_name}_actual_vs_predicted.png'))
    plt.close()

    # Save the model and scaler
    with open(os.path.join(output_directory, f'{project_name}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(output_directory, f'{project_name}_gradient_boosting_model.pkl'), 'wb') as f:
        pickle.dump(gb_model, f)

# Load catalog and create feature dataframe
catalog_file = os.path.join(cat_directory, 'Mars_InSight_training_catalog_final.csv')
catalog = load_catalog(catalog_file)
feature_dataframe = create_feature_dataframe(catalog)

# Train the model and make predictions
train_and_predict_model(feature_dataframe)
