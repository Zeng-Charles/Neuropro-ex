import matplotlib.pyplot as plt
import wfdb
import pandas as pd
import dataprocess
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from cal_features import calculate_features

def extract_windows(data, window_size=100, overlap=0.5):
    """
    extract windows with overlap。
    
    input:
    - data: input data (shape: [num_samples, num_channels])
    - window_size: window szie (100)
    - overlap: overlap of window (50%)

    return:
    - windows:  (shape: [num_windows, window_size, num_channels])
    """
    # calculate step size, how many data in each step
    step_size = int(window_size * (1 - overlap))
    
    # calculate number of windows
    num_samples, num_channels = data.shape
    num_windows = (num_samples - window_size) // step_size + 1
    
    # set windows
    windows = np.zeros((num_windows, window_size, num_channels))
    
    # extract windows
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]
    
    return windows

def process_all_data(data_list):
    """
    process all data and return processed data in 3D。
    
    input: 
    - data_list: data after filter

    return:
    - all_windows: 3D (shape: [num_windows, sampling_points, num_channels])
    """
    all_windows = []
    
    for data in data_list:
        windows = extract_windows(data, window_size=100, overlap=0.5)
        all_windows.append(windows)
    
    all_windows = np.concatenate(all_windows, axis=0)
    
    return all_windows

def compute_correlation_matrix(data):
    """
    compute correlation matrix between each features

    input:
    - data: feature in shape [num_winodws, num_features, num_channels]

    output:
    - correlation_matrix 
    """
    correlation_matrix = []
    for window in data:

        window_correlation = np.corrcoef(window)
        correlation_matrix.append(window_correlation)

    correlation_matrix = np.array(correlation_matrix)
    correlation_matrix = np.mean(correlation_matrix, axis=0)
    return correlation_matrix

def remove_feature(correlation_matrix, threshold = 0.9):
    """
    remove feature with high correlation

    input: correlation_matrix (pd.DataFrame)
           threshold (float) 0.9

    output: corr_reduced, 
            highly_correlated_features, 
            selected_features
    """
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    highly_correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    selected_features = [item for item in correlation_matrix.columns if item not in highly_correlated_features]

    corr_reduced = correlation_matrix.drop(columns=highly_correlated_features, index=highly_correlated_features)
    

    print(f"number of original feature: {correlation_matrix.shape[1]}")
    print(f"number of feature after selection : {corr_reduced.shape[1]}")
    print(f"reduced features: {highly_correlated_features}")
    print(f"seleted features: {selected_features}")

    return corr_reduced, highly_correlated_features, selected_features


def calculate_feature_importance(selected_features, label, feature_names=None, top_n=4):
    """
    calculate feature importance and histogram

    input:
        selected_features (np.array):  (num_windows, num_features, num_channels)。
        num_channels (int): 
        num_features (int): 
        lable (np.array):
        feature_names (list[str]): names of features
        top_n (int): number of importance features

    return:
        global_feature_importances (np.array): 
        top_features_indices (np.array): 
        top_features_importance (np.array): 
    """
    num_windows,num_features,num_channels = selected_features.shape

    global_feature_importances = np.zeros(num_features)

    for channel in range(num_channels):
        X_channel = selected_features[:, :, channel]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_channel)

        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_scaled, label)

        result = permutation_importance(
            model, X_scaled, label, scoring='accuracy', n_repeats=10, random_state=42
        )

        global_feature_importances += result.importances_mean

    global_feature_importances /= num_channels

    top_features_indices = np.argsort(global_feature_importances)[-top_n:][::-1]
    top_features_importance = global_feature_importances[top_features_indices]

    print("Top Features:")
    for i, (index, importance) in enumerate(zip(top_features_indices, top_features_importance)):
        feature_name = feature_names[index] if feature_names is not None else f"Feature {index}"
        print(f"{i+1}. {feature_name}: Importance = {importance:.4f}")

    plot_feature_importance(
        global_feature_importances, 
        top_features_indices, 
        top_features_importance, 
        feature_names
    )

    return global_feature_importances, top_features_indices, top_features_importance


def plot_feature_importance(global_feature_importances, top_features_indices, top_features_importance, feature_names=None):
    """
    plot feature importance histogram

    input：
        global_feature_importances (np.array): 
        top_features_indices (np.array): 
        top_features_importance (np.array): 
        feature_names (list[str]): 
    """
    num_features = len(global_feature_importances)

    # 如果没有提供特征名称，则使用索引
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(num_features)]

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), global_feature_importances, color='lightblue', edgecolor='black')
    plt.bar(top_features_indices, top_features_importance, color='orange', edgecolor='black')

    plt.xlabel('Feature Names')
    plt.ylabel('Importance')
    plt.title('Feature Importance Histogram')
    plt.xticks(range(num_features), feature_names, rotation=45, ha='right')
    plt.legend(['Other Features', 'Top Features'])
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    lowcut = 10.0
    highcut = 500.0
    cutoff_freq = 50.0

    # get all_window in shape [num_window, sampling points, num_channel]
    label_maintenance = np.loadtxt('label_maintenance.txt', delimiter=',')

    pn_dir = 'hd-semg/1.0.0/pr_dataset/subject01_session1'
    num_samples = 9

    data_list = []

    for i in [1, 3]:
        sample_name = f"maintenance_raw_sample{i}"
        record = wfdb.rdrecord(sample_name, pn_dir=pn_dir)
        print(record.p_signal.shape)
        processed_data = dataprocess.dataprocess(record.p_signal, lowcut, highcut, cutoff_freq, record.fs)
        data_list.append(processed_data)  


    all_window = process_all_data(data_list)
    print("all_window.shape:", all_window.shape)

    # plot the first channel of the first window
    # window1_channel1 = all_window[0, :, 0]
    # sampling_rate = 2048  # Hz
    # N = window1_channel1.shape[0]  # num of window
    # time = 1000.00 * np.arange(N) / sampling_rate # ms
    # plt.title("First Window, First Channel Data")
    # plt.xlabel("Time(/ms)")
    # plt.ylabel("Amplitude")
    # plt.plot(time, window1_channel1)
    # plt.show()

    # get features

    feature_names = [
    "zero_crossings",
    "waveform_length",
    "diff_abs_std_dev",
    "integral_absolute_value",
    "log_detector",
    "mean_absolute_value",
    "root_mean_square",
    "absolute_temporal_moment",
    "variance",
    "v_order",
    "mean_frequency",
    "maximum_amplitude",
    "peak_frequency",
    "mean_power",
    "total_power",
    "variance_central_frequency"
    ]

    all_windows_features = calculate_features(all_window, feature_names)

    print("all_windows_features.shape: ",all_windows_features.shape)

    correlation_matrix = compute_correlation_matrix(all_windows_features)
    correlation_matrix = pd.DataFrame(correlation_matrix, columns=feature_names, index=feature_names)
    print(correlation_matrix.shape)

    # plt.figure(figsize=(16, 9)) 
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45, ha='right')
    # plt.show()

    corr_reduced, removed_features_name, selected_features_name = remove_feature(correlation_matrix, threshold=0.9)
    # sns.heatmap(corr_reduced, annot=True, cmap='coolwarm')
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45, ha='right')
    # plt.show()

    selected_features_index = [feature_names.index(name) for name in selected_features_name]
    selected_features = all_windows_features[:, selected_features_index, :]

    print(selected_features.shape)

    label = np.array([1] * 162 + [2] * 162)
    calculate_feature_importance(selected_features,label, selected_features_name)