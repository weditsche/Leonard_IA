#!/usr/bin/env python3
import os
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage

#############################
# Helper Functions
#############################

def load_parameters(param_file):
    config_obj = configparser.ConfigParser()
    config_obj.read(param_file, encoding='utf-8-sig')
    return {
        'output_dir': config_obj.get('input', 'output_dir'),
        'input_dir': config_obj.get('input', 'input_dir'),
        'bleaching_background_threshold': config_obj.getfloat('filter', 'bleaching_background_threshold'),
        'normalize_bleaching': config_obj.getboolean('filter', 'normalize_bleaching'),
        'apply_bleaching_correction': config_obj.getboolean('filter', 'apply_bleaching_correction')
    }

def compute_masked_stats(csv_path):
    """
    Reads the CSV file, selects rows where mask == "combined", and computes, per channel and timepoint,
    the area-weighted mean and weighted standard deviation.
    Returns a dictionary keyed by channel.
    """
    df = pd.read_csv(csv_path)
    df = df[df['mask'] == 'combined'].copy()
    df['timepoint'] = pd.to_numeric(df['timepoint'])
    stats = {}
    for ch in sorted(df['channel'].unique()):
        df_ch = df[df['channel'] == ch]
        grouped = df_ch.groupby('timepoint', group_keys=False).apply(
            lambda g: pd.Series({
                'mean': np.sum(g['mean_intensity'] * g['area']) / np.sum(g['area']),
                'std': np.sqrt(np.sum(g['area'] * (g['mean_intensity'] - (np.sum(g['mean_intensity'] * g['area'])/np.sum(g['area'])))**2) / np.sum(g['area']))
            })
        )
        stats[ch] = grouped
    return stats

def compute_raw_bleaching_stats(input_dir, threshold, normalize):
    """
    Loads the first raw image file and computes, for each timepoint and channel, the mean and std 
    of pixels with intensity greater than the threshold. If none exceed the threshold, uses the entire channel.
    If normalize is True, leaves the values absolute (i.e. no division by timepoint 0).
    Returns a dict with keys for each channel and DataFrames indexed by timepoint.
    """
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.czi', '.tiff', '.tif'))]
    if not input_files:
        print("No raw input files found for bleaching curve.")
        return None
    first_file = os.path.join(input_dir, input_files[0])
    print(f"Computing raw bleaching stats from: {first_file}")
    img_obj = AICSImage(first_file)
    raw_data = img_obj.get_image_data("TCYX")  # shape: (T, C, H, W)
    T, C, H, W = raw_data.shape
    stats = {}
    for ch in range(C):
        means = []
        stds = []
        for t in range(T):
            data = raw_data[t, ch, :, :]
            valid = data[data > threshold]
            if valid.size > 0:
                m = np.mean(valid)
                s = np.std(valid)
            else:
                m = np.mean(data)
                s = np.std(data)
            means.append(m)
            stds.append(s)
        means = np.array(means)
        stds = np.array(stds)
        # For raw bleaching, we do NOT normalize the curve for plotting purposes.
        stats[ch] = pd.DataFrame({'mean': means, 'std': stds}, index=np.arange(T))
    return stats

def error_on_ratio(m1, s1, m0, s0):
    """Error propagation for ratio r = m1/m0."""
    return np.sqrt((s1/m0)**2 + ((m1*s0)/(m0**2))**2)

def plot_with_errorbars(x, y, yerr, xlabel, ylabel, title, filename, output_dir):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

#############################
# Plotting Functions
#############################

def plot_masked_intensities(stats_masked, output_dir):
    # Plot masked intensities for each channel in separate figures.
    for ch, df in stats_masked.items():
        color = "red" if ch == 0 else "green"
        plot_with_errorbars(df.index, df['mean'], df['std'],
                            "Time", f"Weighted Mean Intensity (Ch{ch}, Masked)",
                            f"Masked Intensities vs Time (Channel {ch})",
                            f"plot_masked_intensity_ch{ch}.png", output_dir)

def plot_raw_bleaching(raw_stats, output_dir):
    # Plot raw bleaching curves for each channel.
    for ch, df in raw_stats.items():
        color = "red" if ch == 0 else "green"
        plot_with_errorbars(df.index, df['mean'], df['std'],
                            "Time", f"Raw Intensity (Ch{ch})",
                            f"Raw Bleaching Curve vs Time (Channel {ch})",
                            f"plot_raw_bleaching_ch{ch}.png", output_dir)

def plot_relative_intensity(stats_masked, output_dir):
    # Relative intensity (uncorrected): ratio = (masked intensity ch1) / (masked intensity ch0)
    df0 = stats_masked.get(0)
    df1 = stats_masked.get(1)
    x = np.array(df0.index)
    ratio = df0['mean'] / df1['mean']
    ratio_err = []
    for t in x:
        r_err = error_on_ratio(df1.loc[t, 'mean'], df1.loc[t, 'std'],
                                df0.loc[t, 'mean'], df0.loc[t, 'std'])
        ratio_err.append(r_err)
    ratio_err = np.array(ratio_err)
    plot_with_errorbars(x, ratio, ratio_err,
                        "Time", "Relative Intensity (Ch0/Ch1)",
                        "Relative Intensity vs Time (Uncorrected)",
                        "plot_relative_intensity.png", output_dir)

def plot_bleaching_corrected_relative_intensity(stats_masked, raw_stats, output_dir):
    # Bleaching correction: For each channel, compute correction factor = raw[0] / raw[t].
    df0 = stats_masked.get(0)
    df1 = stats_masked.get(1)
    x = np.array(df0.index)
    corrected0 = []
    corrected1 = []
    for t in x:
        if raw_stats[0].loc[t, 'mean'] != 0 and raw_stats[1].loc[t, 'mean'] != 0:
            # Multiply masked intensity by (raw at time 0)/(raw at time t)
            c0 = df0.loc[t, 'mean'] * (raw_stats[0].loc[0, 'mean'] / raw_stats[0].loc[t, 'mean'])
            c1 = df1.loc[t, 'mean'] * (raw_stats[1].loc[0, 'mean'] / raw_stats[1].loc[t, 'mean'])
        else:
            c0, c1 = 0, 0
        corrected0.append(c0)
        corrected1.append(c1)
    corrected0 = np.array(corrected0)
    corrected1 = np.array(corrected1)
    # Compute the ratio of corrected intensities.
    ratio_corr = corrected0 / corrected1
    ratio_corr_err = []
    for i, t in enumerate(x):
        m0 = corrected0[i]
        s0 = df0.loc[t, 'std'] / raw_stats[0].loc[t, 'mean'] if raw_stats[0].loc[t, 'mean'] != 0 else 0
        m1 = corrected1[i]
        s1 = df1.loc[t, 'std'] / raw_stats[1].loc[t, 'mean'] if raw_stats[1].loc[t, 'mean'] != 0 else 0
        ratio_corr_err.append(error_on_ratio(m1, s1, m0, s0))
    ratio_corr_err = np.array(ratio_corr_err)
    plot_with_errorbars(x, ratio_corr, ratio_corr_err,
                        "Time", "Bleaching-Corrected Relative Intensity (Ch0/Ch1)",
                        "Bleaching-Corrected Relative Intensity vs Time",
                        "plot_relative_intensity_bleaching_corrected.png", output_dir)

def plot_combined_masked_intensities(stats_masked, output_dir):
    # Plot combined masked intensities for channels 0 and 1 on one graph.
    df0 = stats_masked.get(0)
    df1 = stats_masked.get(1)
    plt.figure()
    plt.errorbar(df0.index, df0['mean'], yerr=df0['std'], fmt='-o', capsize=5, color="red", label='Channel 0')
    plt.errorbar(df1.index, df1['mean'], yerr=df1['std'], fmt='-o', capsize=5, color="green", label='Channel 1')
    plt.xlabel("Time")
    plt.ylabel("Weighted Mean Intensity (Masked)")
    plt.title("Combined Masked Intensities vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plot_combined_masked_intensities.png"))
    plt.close()

def plot_bleaching_corrected_combined_intensities(stats_masked, raw_stats, output_dir):
    # Correct masked intensities using bleaching correction factor and plot combined.
    df0 = stats_masked.get(0)
    df1 = stats_masked.get(1)
    x = np.array(df0.index)
    corrected0 = []
    corrected1 = []
    for t in x:
        if raw_stats[0].loc[t, 'mean'] != 0 and raw_stats[1].loc[t, 'mean'] != 0:
            corr0 = df0.loc[t, 'mean'] * (raw_stats[0].loc[0, 'mean'] / raw_stats[0].loc[t, 'mean'])
            corr1 = df1.loc[t, 'mean'] * (raw_stats[1].loc[0, 'mean'] / raw_stats[1].loc[t, 'mean'])
        else:
            corr0, corr1 = 0, 0
        corrected0.append(corr0)
        corrected1.append(corr1)
    corrected0 = np.array(corrected0)
    corrected1 = np.array(corrected1)
    plt.figure()
    plt.errorbar(x, corrected0, fmt='-o', capsize=5, color="red", label='Channel 0 Corrected')
    plt.errorbar(x, corrected1, fmt='-o', capsize=5, color="green", label='Channel 1 Corrected')
    plt.xlabel("Time")
    plt.ylabel("Bleaching-Corrected Masked Intensity")
    plt.title("Bleaching-Corrected Combined Masked Intensities vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plot_bleaching_corrected_combined_masked_intensities.png"))
    plt.close()

#############################
# Main Function
#############################

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(base_dir, "..", "parameters.txt")
    PARAMS = load_parameters(param_file)
    OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, PARAMS['output_dir']))
    INPUT_DIR = os.path.abspath(os.path.join(base_dir, PARAMS['input_dir']))
    
    csv_path = os.path.join(OUTPUT_DIR, "intensity_analysis.csv")
    
    # Compute masked statistics from CSV.
    stats_masked = compute_masked_stats(csv_path)
    
    # Compute raw bleaching statistics from the first raw file (absolute values).
    raw_stats = compute_raw_bleaching_stats(INPUT_DIR, PARAMS['bleaching_background_threshold'], normalize=False)
    
    # Plot 1 & 2: Masked intensities for channels 0 and 1.
    plot_masked_intensities(stats_masked, OUTPUT_DIR)
    
    # Plot 3: Raw bleaching curves.
    if raw_stats is not None:
        plot_raw_bleaching(raw_stats, OUTPUT_DIR)
    
    # Plot 4: Relative intensity (un-corrected) between channels (Ch1 / Ch0).
    plot_relative_intensity(stats_masked, OUTPUT_DIR)
    
    # Plot 5: Bleaching-corrected relative intensity between channels.
    if raw_stats is not None:
        plot_bleaching_corrected_relative_intensity(stats_masked, raw_stats, OUTPUT_DIR)
    
    # Plot 6: Combined masked intensities for both channels.
    plot_combined_masked_intensities(stats_masked, OUTPUT_DIR)
    
    # Plot 7: Bleaching-corrected combined masked intensities.
    if raw_stats is not None:
        plot_bleaching_corrected_combined_intensities(stats_masked, raw_stats, OUTPUT_DIR)
    
    print(f"All plots have been saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
