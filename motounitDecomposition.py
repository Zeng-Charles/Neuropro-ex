import scipy
import pickle
import altair as alt

import numpy as np
import matplotlib.pyplot as plt

from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *


# Load the .mat file
data = scipy.io.loadmat('/Users/a1/Desktop/EX2/Experimental_data_Raw/GL_10.mat')

# Access the variables in the .mat file
print(data.keys())

# Extract the variables
SIG = data['SIG']
ref_signal = data['ref_signal']
fsamp = data['fsamp']

print(SIG.shape)
print(ref_signal.shape)
print(fsamp.shape)

# Concatenate all non-empty channels of the EMG signal
emg_data = np.vstack([channel for row in SIG for channel in row if channel.size > 0])

time = np.arange(emg_data.shape[1]) / fsamp[0, 0]  # Convert samples to time in seconds

plt.figure(figsize=(10, 20))  # Increase figure size for better readability

for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / 10))  # Use a color map for variety
    plt.title(f'Channel {i + 1}', fontsize=10)
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Amplitude', fontsize=8)
    plt.grid(True)  # Enable grid for better visibility
    plt.legend(loc='upper right', fontsize=8)  # Add a legend for clarity

plt.tight_layout()
# plt.show()

#Decompose the EMG signal
output = decomposition(
    SIG,
    discard=5,
    R=16,
    M=64,
    bandpass=True,
    lowcut=10,
    highcut=900,
    fs=2048,
    order=6,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter_sep=10,
    l=31,
    sil_pnr=True,
    thresh=0.9,
    max_iter_ref=10,
    random_seed=None,
    verbose=False
)

# Save the output
decomp_GL_10 = output 
decomp_gl_10_pkl = open('decomp_GL_10_pkl.obj', 'wb') 
pickle.dump(decomp_GL_10, decomp_gl_10_pkl)

# Load the output
# with open('decomp_sample_pkl.obj', 'rb') as f: output = pickle.load(f)

# visualise the decomposition
# visualize_decomp(output, emg_data)