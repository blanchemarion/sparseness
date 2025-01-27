import pandas as pd
import numpy as np
from functions import load_dataset
import matplotlib.pyplot as plt
import pickle


# Kernel Computation with Pulse-Centered Averaging
# Kernel computed as follows: 
# - Assign unique identifiers to odorant pulses
# - Extend them by a fixed time window to capture delayed calcium responses
# - For each pulse, relevant features such as time, phase, valve strength, and adjusted calcium amplitude are extracted and aggregated within the pulse window


def max_interval(series, start_index, end_index):
    return series.iloc[start_index:end_index].max()

def compute_kernels(calcium_pulse, n_bins):
    min_phase, max_phase = calcium_pulse['phase'].min(), calcium_pulse['phase'].max()
    bin_edges = np.linspace(min_phase, max_phase, n_bins + 1)     
    bin_indices = np.digitize(calcium_pulse['phase'], bin_edges) - 1

    sniff_kernel = np.zeros(n_bins)  # Initialize kernel with zeros

    for j in range(n_bins):
        indices_in_bin = np.where(bin_indices == j)[0]
        if len(indices_in_bin) > 0:
            sniff_kernel[j] = np.nanmean(calcium_pulse['amp_ca'][indices_in_bin])
        else:
            sniff_kernel[j] = np.nan  # Assign NaN or any other placeholder for empty bins
    
    return sniff_kernel


def plot_kernel(calcium_pulse_dict, n_bins=20):
    # Plot of the estimated kernel for each sniff phase
    fig, axs = plt.subplots(ncols=len(calcium_pulse_dict), figsize=(12, 3), sharex=True)

    for i, (key, calcium_pulse) in enumerate(calcium_pulse_dict.items()):
        kernels = compute_kernels(calcium_pulse, n_bins)

        min_phase, max_phase = calcium_pulse_dict[key]['phase'].min(), calcium_pulse_dict[key]['phase'].max()
        phase_edges = np.linspace(min_phase, max_phase, n_bins + 1)
        phase_midpoints = (phase_edges[:-1] + phase_edges[1:]) / 2

        axs[i].plot(phase_midpoints, kernels, color='skyblue')
        axs[i].set_title(f'Sniff Kernel - {key}', fontsize=9)
        axs[i].set_xlabel('Sniff Phase (radians)', fontsize=9)
        axs[i].set_ylabel('Average Calcium Amplitude', fontsize=9)

    fig.tight_layout()
    plt.show()

def assign_pulse_ids(ca_pulse, window=900):
    valve_on = ca_pulse['valve'] != 0

    pulse_starts = np.where(valve_on & ~valve_on.shift(fill_value=False))[0]
    pulse_ends = np.where(~valve_on & valve_on.shift(fill_value=False))[0]

    ca_pulse['pulse_id'] = np.nan
    for pulse_idx, (start, end) in enumerate(zip(pulse_starts, pulse_ends), start=1):
        window_end = min(end + window, len(ca_pulse) - 1)
        ca_pulse.loc[start:window_end, 'pulse_id'] = pulse_idx

    return ca_pulse.dropna(subset=['pulse_id'])

def process_pulse_data(ca_pulse):
    return (
        ca_pulse.groupby('pulse_id')
        .agg(
            time=('time', 'first'),
            phase=('phase', lambda x: x.head(50).mean()),  # Mean of first 50 samples during pulse
            valve=('valve', 'first'),
            amp_ca=('avg_roi_ca', lambda x: max_interval(x, 200, 600))# - abs(max_interval(x, 50, 200)))
        )
        .reset_index(drop=True)
    )

def main():
    # Load the dataset
    with open("Data/animals_data_processed.pkl", "rb") as f:
        data = pickle.load(f)

    calcium_pulse_dict = {}
    for key, ca_interp in data["ca_interp_dict"].items():
        ca_pulse = pd.DataFrame({
            'time': data["t_breath"][key],
            'phase': data["phase_peaks_dict"][key],
            'valve': data["valve_dict"][key],
            'avg_roi_ca': ca_interp.mean(axis=1)
        })

        ca_pulse = assign_pulse_ids(ca_pulse)
        calcium_pulse_dict[key] = process_pulse_data(ca_pulse)

    plot_kernel(calcium_pulse_dict)

if __name__ == "__main__":
    main()


