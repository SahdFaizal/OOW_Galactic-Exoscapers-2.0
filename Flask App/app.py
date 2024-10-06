from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from scipy.signal import find_peaks
from obspy import read
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

def load_model_and_scaler(model_choice):
    if model_choice == 'mars':
        model_path = 'static/resources/mars_gradient_boosting_model.pkl'
        scaler_path = 'static/resources/mars_scaler.pkl'
    else:
        model_path = 'static/resources/best_random_forest_model.pkl'
        scaler_path = 'static/resources/scaler.pkl'

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    except FileNotFoundError as e:
        print("File not found:", e)
        raise

    return model, scaler

def load_seismic_data(file_stream):
    try:
        return read(file_stream)
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

def plot_seismic_data_with_prediction(tr, predicted_time):
    fs = tr.stats.sampling_rate
    time_array = np.arange(len(tr.data)) / fs

    plt.figure(figsize=(10, 6))
    plt.plot(time_array, tr.data, label='Seismic Velocity (m/s)', color='black')
    plt.axvline(x=predicted_time, color='red', linestyle='--', label='Predicted Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Seismic Data with Prediction')
    plt.legend()

    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', messages=["No file part"])

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', messages=["No selected file"])

    model_choice = request.form.get('model')

    try:
        model, scaler = load_model_and_scaler(model_choice)
    except Exception as e:
        return render_template('index.html', messages=["Model loading failed."])

    if file and file.filename.endswith('.mseed'):
        try:
            seismic_data = load_seismic_data(file)

            if seismic_data is None:
                return render_template('index.html', messages=["Failed to load seismic data."])

            features = extract_velocity_features(seismic_data[0])
            feature_df = pd.DataFrame([features])
            scaled_features = scaler.transform(feature_df)
            predictions = model.predict(scaled_features)

            # Get the predicted time
            predicted_time = predictions[0]

            # Plot the seismic data with the predicted time
            plot_image = plot_seismic_data_with_prediction(seismic_data[0], predicted_time)

            return render_template('index.html', predicted_time=predicted_time, plot_image=plot_image)

        except Exception as e:
            print(f"Error processing file: {e}")
            return render_template('index.html', messages=[str(e)])
    else:
        return render_template('index.html', messages=["File format not supported. Please upload a .mseed file."])

