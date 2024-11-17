import numpy as np

# 定义各个特征计算函数
def zero_crossings(window, threshold=0.005):
    return ((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > threshold)).sum()

def waveform_length(window):
    return np.sum(np.abs(np.diff(window)))

def diff_abs_std_dev(window):
    return np.std(np.abs(np.diff(window)))

def integral_absolute_value(window):
    return np.sum(np.abs(window))

def log_detector(window):
    return np.exp(np.mean(np.log(np.abs(window) + 1e-10)))

def mean_absolute_value(window):
    return np.mean(np.abs(window))

def root_mean_square(window):
    return np.sqrt(np.mean(window ** 2))

def absolute_temporal_moment(window):
    return np.mean(np.abs(window)**2)

def variance(window):
    return np.var(window)

def v_order(window, order=3):
    return np.mean(np.abs(window) ** order)

def mean_frequency(window):
    freq_domain = np.fft.fft(window)
    magnitudes = np.abs(freq_domain)
    frequencies = np.fft.fftfreq(len(window))
    return np.sum(frequencies * magnitudes) / np.sum(magnitudes)

def maximum_amplitude(window):
    return np.max(np.abs(window))

def peak_frequency(window):
    freq_domain = np.fft.fft(window)
    magnitudes = np.abs(freq_domain)
    frequencies = np.fft.fftfreq(len(window))
    peak_idx = np.argmax(magnitudes)
    return frequencies[peak_idx]

def mean_power(window):
    return np.mean(np.abs(window) ** 2)

def total_power(window):
    return np.sum(np.abs(window) ** 2)

def variance_central_frequency(window):
    freq_domain = np.fft.fft(window)
    magnitudes = np.abs(freq_domain)
    frequencies = np.fft.fftfreq(len(window))
    mean_freq = mean_frequency(window)
    return np.sum(magnitudes * (frequencies - mean_freq) ** 2) / np.sum(magnitudes)

feature_functions = {
    "zero_crossings": zero_crossings,
    "waveform_length": waveform_length,
    "diff_abs_std_dev": diff_abs_std_dev,
    "integral_absolute_value": integral_absolute_value,
    "log_detector": log_detector,
    "mean_absolute_value": mean_absolute_value,
    "root_mean_square": root_mean_square,
    "absolute_temporal_moment": absolute_temporal_moment,
    "variance": variance,
    "v_order": v_order,  
    "mean_frequency": mean_frequency,
    "maximum_amplitude": maximum_amplitude,
    "peak_frequency": peak_frequency,
    "mean_power": mean_power,
    "total_power": total_power,
    "variance_central_frequency": variance_central_frequency
}


def calculate_features(emg_data, features_names):
    """
    emg_data : [num_windows, sanmpling_points , num_channels]
    name_features: name of features
    """
    all_window_features = []

    for window in emg_data:
        window_feature = []

        for channel in window.T:
            channel_features = []

            for feature in features_names:
                func = feature_functions[feature]

                if feature == "v_order":  
                    features_values = func(channel, order=3)
                else:
                    features_values = func(channel)

                channel_features.append(features_values)

            window_feature.append(channel_features)
        all_window_features.append(window_feature)

    return np.array(all_window_features).transpose(0, 2, 1)
