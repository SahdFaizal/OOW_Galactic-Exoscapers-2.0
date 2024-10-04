import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from obspy import read
import pickle
from scipy.signal import find_peaks

cat_directory = './data/lunar/training/catalogs/'
data_directory = './data/lunar/training/data/S12_GradeA/'
output_directory = './Resources/lunar'

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
    print("Catalog columns:", cat.columns.tolist())
    
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

def train_random_forest_model(feature_data):
    X = feature_data.drop(columns=['time_rel(sec)', 'filename'])
    y = feature_data['time_rel(sec)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    # Using RandomizedSearchCV
  # Using GridSearchCV with cv=5
    grid_search = GridSearchCV(rf_model, param_grid, cv=5,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_rf_model = grid_search.best_estimator_

    y_pred_train = best_rf_model.predict(X_train_scaled)
    y_pred_test = best_rf_model.predict(X_test_scaled)

    mse_train = mean_absolute_error(y_train, y_pred_train)
    mse_test = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"Training Mean Absolute Error: {mse_train}")
    print(f"Testing Mean Absolute Error: {mse_test}")
    print(f"R^2 Score: {r2}")

    train_results = pd.DataFrame({'Filename': feature_data.loc[X_train.index, 'filename'], 'Actual': y_train, 'Predicted': y_pred_train})
    test_results = pd.DataFrame({'Filename': feature_data.loc[X_test.index, 'filename'], 'Actual': y_test, 'Predicted': y_pred_test})

    train_results.to_csv(os.path.join(output_directory, 'train_predictions.csv'), index=False)
    test_results.to_csv(os.path.join(output_directory, 'test_predictions.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_pred_train, alpha=0.7, label='Training Data', color='blue')
    plt.scatter(y_test, y_pred_test, alpha=0.7, label='Testing Data', color='orange')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values (Training and Testing Data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'actual_vs_predicted_training_testing_rf.png'))
    plt.close()

    feature_importance = best_rf_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(20, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(output_directory, 'feature_importance_rf.png'))
    plt.close()

    with open(os.path.join(output_directory, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(output_directory, 'best_random_forest_model.pkl'), 'wb') as f:
        pickle.dump(best_rf_model, f)

# Load catalog and create feature dataframe
catalog_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')
catalog = load_catalog(catalog_file)
feature_dataframe = create_feature_dataframe(catalog)

# Train the model
best_model = train_random_forest_model(feature_dataframe)
