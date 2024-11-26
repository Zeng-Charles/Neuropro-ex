import numpy as np
import wfdb
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import dataprocess
from FeatureExtractionSelection import process_all_data
from FeatureExtractionSelection import compute_force_labels
from cal_features import calculate_features

if __name__ == "__main__":

    label_maintenance = np.loadtxt('label_maintenance.txt', delimiter=',')
    pn_dir = 'hd-semg/1.0.0/1dof_dataset/subject01_session1'

    sample_name = '1dof_raw_finger1_sample1'
    emg = wfdb.rdrecord(sample_name, pn_dir=pn_dir)
    emg_data = emg.p_signal

    label_name = '1dof_force_finger1_sample1'
    record = wfdb.rdrecord(label_name, pn_dir=pn_dir)
    force_data = record.p_signal[:, 0]

    # upsampling
    original_time = np.linspace(0 ,1, num=force_data.shape[0])
    target_time = np.linspace(0, 1, num=emg_data.shape[0])
    interp_func = interp1d(original_time, force_data, axis=0, kind='cubic') #'linear' / 'cubic'
    upsampled_force = interp_func(target_time)

    # print(upsampled_force.shape)
    # plt.plot(original_time, force_data[:, 0], label="Original Data", marker='o')
    # plt.plot(target_time, upsampled_force[:, 0], label="Interpolated Data")
    # plt.legend()
    # plt.show()

    # filter data
    lowcut = 10.0
    highcut = 500.0
    cutoff_freq = 50.0
    processed_data = dataprocess.dataprocess(emg_data, lowcut, highcut, cutoff_freq, emg.fs)

    # extract windows
    data_list = [processed_data]
    all_window = process_all_data(data_list)
    print("all_window.shape:", all_window.shape)

    force_label = compute_force_labels(upsampled_force)
    print("force_labels.shape:", force_label.shape) 

    selected_feature_names = [
        "waveform_length",
        "variance_central_frequency",
        "zero_crossings",
        "maximum_amplitude"
    ]

    all_windows_features = calculate_features(all_window, selected_feature_names)
    features_flattened = all_windows_features.reshape(all_windows_features.shape[0],-1)

    
    # 1. 将数据拆分为训练集和测试集
    train_data, test_data, train_label, test_label = train_test_split(features_flattened, force_label, train_size=0.7, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # 2. 线性回归模型
    linear_regressor = LinearRegression()
    linear_regressor.fit(train_data, train_label)
    label_pred_linear = linear_regressor.predict(test_data)

    # 3. 支持向量回归 (RBF 核)
    svr_regressor = SVR(kernel='rbf')
    svr_regressor.fit(train_data, train_label)
    label_pred_svr = svr_regressor.predict(test_data)

    # 4. 计算 MSE
    mse_linear = mean_squared_error(test_label, label_pred_linear)
    mse_svr = mean_squared_error(test_label, label_pred_svr)

    print(f"- Linear Regression MSE: {mse_linear:.4f}")
    print(f"- SVR (RBF Kernel) MSE: {mse_svr:.4f}")  

    # 可视化结果
    plt.figure(figsize=(10, 6))

    # 绘制真实值和预测值
    plt.plot(test_label, label='Ground Truth', color='black', linestyle='--')
    plt.plot(label_pred_linear, label='Linear Regression Prediction', color='blue')
    plt.plot(label_pred_svr, label='SVR Prediction', color='red')

    plt.title(f"Force Prediction")
    plt.xlabel('Sample Index')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)

    plt.show()

