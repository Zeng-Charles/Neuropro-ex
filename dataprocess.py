import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

def bandpass_filter(data, lowcut, highcut, fs, order=8):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low,high], btype='band')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def notch_filter(data, cutoff_freq, fs, quality_factor=30):
    nyquist = 0.5*fs
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.iirnotch(normalized_cutoff, quality_factor)
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def plot_frequency_spectrum(signal, fs, title="Frequency Spectrum"):
    if isinstance(signal, pd.Series):
        signal = pd.to_numeric(signal,errors='coerce').fillna(0).values.astype(float) #if data is pd series
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.fft.fft(signal)
    plt.plot(freqs[:n // 2], np.abs(fft_vals[:n // 2]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def dataprocess(data, lowcut, highcut, cutoff_freq, fs):
    processed_data= np.zeros_like(data)

    for channel in range(data.shape[1]):
        channel_data = data[:, channel]

        bandpassed_data = bandpass_filter(channel_data, lowcut, highcut, fs)

        notchfiltered_data = notch_filter(bandpassed_data, cutoff_freq,fs)

        processed_data[:, channel] = notchfiltered_data
    
    return processed_data

if __name__ == "__main__":
    #read data
    emg_data = pd.read_csv('extension_data.csv')
    print(emg_data.shape)

    # get first channel
    emg_channel1 = emg_data.iloc[0]
    sampling_rate = 2000  # 2000 Hz
    N = len(emg_channel1)
    T = 1.0 / sampling_rate
    time = np.arange(N) * T

    lowcut = 10.0
    highcut = 500
    bandpass_filtered_data = pd.DataFrame(index=emg_data.index, columns=emg_data.columns)
    for index in range(emg_data.shape[0]):
        bandpass_filtered_data.iloc[index] = bandpass_filter(emg_data.iloc[index],lowcut, highcut, sampling_rate)
    # filtedData = data.apply(lambda x: bandpass_filter(x.values, lowcut, highcut, sampling_rate), axis=1)
    # filtedData = pd.DataFrame(filtedData.tolist())
    notch_freq = 50.0
    notch_filtered_data = pd.DataFrame(index=bandpass_filtered_data.index, columns=bandpass_filtered_data.columns)
    for index in range(bandpass_filtered_data.shape[0]):
        notch_filtered_data.iloc[index] = notch_filter(bandpass_filtered_data.iloc[index], notch_freq, sampling_rate)
    print(notch_filtered_data.shape)


    plt.figure(num="Frequency_spectrum")
    plt.subplot(2,1,1)
    plot_frequency_spectrum(emg_channel1, sampling_rate, "EMG_Channel1_Raw_Frequency_spectrum")
    plt.subplot(2,1,2)
    plot_frequency_spectrum(notch_filtered_data.iloc[0], sampling_rate, "EMG_Channel1_filtered_Frequency_spectrum")
    plt.subplots_adjust(hspace=0.5)

    plt.figure(num="ENG_Value")  
    plt.plot(time, emg_channel1)
    plt.title("EMG_Channel1_Raw")
    plt.xlabel("Time (seconds)")  
    plt.ylabel("EMG Value")  
    plt.grid(True) 
    plt.show()




