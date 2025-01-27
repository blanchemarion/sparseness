import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from functions import load_dataset

breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, phase_hilbert_dict, phase_peaks_dict, t_breath, t_pulse, t_calcium, animals = load_dataset(path="Processed/all_animals_data.pkl")

stim = valve_dict['HW1']/100
resp = calcium_dict['HW1'].mean(axis=1).to_numpy()
phase = phase_peaks_dict['HW1']

stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)
interp_func_prev = interp1d(
    t_calcium[0], 
    calcium_dict['HW1'].mean(axis=1).to_numpy(), 
    kind="previous", 
    bounds_error=False, 
    fill_value="extrapolate")
nn_interp = interp_func_prev(stim_ts)

interp_func_lin = interp1d(
    t_calcium[0], 
    calcium_dict['HW1'].mean(axis=1).to_numpy(), 
    kind="linear", 
    bounds_error=False, 
    fill_value="extrapolate")
nn_interp_lin = interp_func_lin(stim_ts)

plt.figure(figsize=(12, 4))
plt.plot(stim_ts[0:10000], nn_interp[0:10000], label='Previous nn Ca', color='blue', alpha=0.2)
plt.scatter(resp_ts[0:100], resp[0:100], label='Orginal Ca', color='red', alpha=0.7)
plt.plot(stim_ts[0:10000], ca_interp_dict['HW1'].mean(axis=1).to_numpy()[0:10000], label='Linear interp', color='green', alpha=0.2)
plt.title('Comparison of interpolation methods')
plt.xlabel('Time (ms)')
plt.ylabel('Signal')
plt.legend()
plt.show()


"""window_size = 0.4 # seconds

stim_agg = np.zeros_like(resp_ts)

for i, t in enumerate(resp_ts):
    start_time = t - window_size
    valid_indices = np.where((stim_ts >= start_time) & (stim_ts < t))[0]
    
    if len(valid_indices) > 0:
        stim_agg[i] = np.sum(stim[valid_indices])
    else:
        stim_agg[i] = 0  
"""
