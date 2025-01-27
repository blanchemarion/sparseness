from functions import load_dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""with open("predicted_hw1_voxel.pkl", "rb") as f:
    predicted_resp = pickle.load(f)

breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, instant_phase_dict, t_breath, _, _, animals = load_dataset()
resp = calcium_dict['HW1'].mean(axis=1).to_numpy()
stim = valve_dict['HW1']

stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

plt.figure(figsize=(20, 6))
plt.plot(stim_ts[500000:600000], predicted_resp[500000:600000], label='Predicted')
plt.plot(resp_ts[5000:6000], resp[5000:6000], label='Actual')
plt.legend()
plt.show()"""

"""with open("kernel_hw1_voxel.pkl", "rb") as f :
    kernel = pickle.load(f)
with open("kernel_timestamps_hw1_voxel.pkl", "rb") as f :
    kernel_timestamps = pickle.load(f)

plt.figure(figsize=(20, 6))
plt.plot(kernel_timestamps, kernel)
plt.legend()
plt.show()
"""

breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, instant_phase_dict, t_breath, _, _, animals = load_dataset()

with open("voxel_kernel_dict.pkl","rb") as f:
    f_hats_dict = pickle.load(f)

with open("voxel_kernel_timestamps_dict.pkl","rb") as f:
    f_hats_ts_dict = pickle.load(f)

with open("predicted_voxel_dict.pkl","rb") as f:
    predicted_resp_dict=pickle.load(f)


for i, key in enumerate(f_hats_dict.keys()):
    plt.figure(figsize=(10, 3))
    t=np.arange(-(len(f_hats_dict[key])), 0)
    plt.plot(t, f_hats_dict[key])
    plt.legend()
    plt.title(f'Kernel for {key} using Voxel Timing and Reverse Correlation')
    plt.show()

for i, key in enumerate(predicted_resp_dict.keys()):
    resp = calcium_dict[key].mean(axis=1).to_numpy()
    stim = valve_dict[key]

    stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
    resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

    plt.figure(figsize=(10, 3))
    plt.plot(resp_ts[0:100], resp[0:100], label='Actual Calcium Response (Normalized)', color='blue', alpha=0.7)
    plt.plot(stim_ts[0:10000], predicted_resp_dict[key], label='Predicted Response (Normalized)', color='red', linestyle='--', alpha=0.7)
    plt.title(f'Normalized Actual vs. Predicted Calcium Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
