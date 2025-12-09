import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

import warnings
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

np.random.seed(42)

warnings.filterwarnings('ignore')

data_type = 'male_high_rank'
CONFIG = {
    'data_paths': {
        'dataset_name': 'DA_datasets.pkl',
        'continuous_dataset_name': 'DA_continuous_datasets.pkl',
        'input_directory': f'./data/original_data/{data_type}',
        'datasets_directory': f'./data/reorganized_dataset/{data_type}',
        'plots_directory': f'./figs/preprocessing_DA_dataset/{data_type}',
    },
    'experimental_setup': {
        'channels': ['410', '470'],
        'brain_regions': {'CH1': 'mPFC', 'CH2': 'NAc'},
        'sampling_rate': 50,
    },
    'signal_processing': {
        'minimum_stable_length': 9000,
        'median_kernel_size': 7,
        'lowpass_cutoff': 10,
        'baseline_cutoff': 0.001, # 5.5 min
        'highpass_cutoff': 0.001,
        'gaussian_sigma': 5,
        'outlier_window_minutes': 100,
        'outlier_threshold_factor': 15,
        'outlier_expand_seconds': 3,
        'outlier_gap_seconds': 3,
    },
    'analysis_parameters': {
        'scatter_sample_size': 2000,
        'plot_resolution': 300,
        'intersection_threshold': 3000,  # 50*60*1, 1 min
    },
    'processing_options': {
        'generate_plots': False,
        'enable_multiprocessing': True,
        'max_workers': None,  # None uses cpu_count() // 2
        'target_mice': [],
        'exclude_mice': [],
        # If True, compute baseline and 410->470 regression per trial instead of using a global baseline
        # This will segment the data by trials parsed from behavior tags and perform the offset/baseline
        # and motion correction within each trial. When False, original global method is used.
        'use_per_trial_baseline': False,
        # When False (legacy path), this controls whether to fit 410->470 per-trial
        # (True) or fit a single global 410->470 regression (False).
        'per_trial_regression': False,
        'target_phases': ['baseline', 'win-1', 'win-2', 'win-3', 'win-4'],
        'target_channels': ['CH1','CH2']
    }
}

plt.rcParams.update({
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

def get_mice_ids():
    data_directory = CONFIG['data_paths']['input_directory']
    mice_directories = [d for d in os.listdir(data_directory)]
    available_mice = sorted(mice_directories)
    target_mice = CONFIG['processing_options']['target_mice']
    exclude_mice = CONFIG['processing_options']['exclude_mice']
    if target_mice:
        mice_to_process = [mouse for mouse in target_mice if mouse in available_mice]
    elif exclude_mice:
        mice_to_process = [mouse for mouse in available_mice if mouse not in exclude_mice]
    else:
        mice_to_process = available_mice
    return mice_to_process

def get_filter_coefficients(cutoff, order, filter_type, sampling_rate):
    """Cache filter coefficients to avoid recalculation."""
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist
    return signal.butter(order, normalized_cutoff, btype=filter_type)


def linregress_no_nan(x, y):
    """Perform linear regression excluding NaN/inf values. Returns tuple like
    scipy.stats.linregress. If there are fewer than 3 valid points or an
    exception occurs, returns zeros for slope/intercept/r/p/stderr to match
    existing fallback behavior.
    """
    try:
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return linregress(x[mask], y[mask])
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0

def detect_signal_segments(fluorescence_signal, time_points, behavior_tags):
    """Detect signal segments for offset and background calculations."""
    # Find transition point using signal difference
    signal_diff = np.diff(fluorescence_signal)
    cumulative_window_size = 1000 # 50*20, 20 seconds
    cumulative_difference = np.convolve(signal_diff, np.ones(cumulative_window_size), mode='valid')
    transition_index = np.argmax(cumulative_difference)

    # Get background period from behavior tags
    behavior_t = behavior_tags[behavior_tags['Behavior'] == 't']
    background_start_time = behavior_t['Start (s)'].iloc[0]
    background_end_time = behavior_t['Start (s)'].iloc[-1]

    background_start_index = np.argmin(np.abs(time_points - background_start_time))
    background_end_index = np.argmin(np.abs(time_points - background_end_time))

    # Find stable offset region
    offset_signal = fluorescence_signal[:transition_index]
    min_stable = CONFIG['signal_processing']['minimum_stable_length']

    smoothed = gaussian_filter1d(offset_signal, sigma=CONFIG['signal_processing']['gaussian_sigma'])
    window_size = min(min_stable, len(offset_signal) // 3)

    variances = np.array([
        np.var(smoothed[i:i + window_size])
        for i in range(0, len(offset_signal) - window_size + 1, window_size // 4)
    ])
    best_idx = np.argmin(variances) * (window_size // 4)
    offset_start_index = max(0, best_idx)
    offset_end_index = min(transition_index, best_idx + window_size)

    return {
        'transition_index': transition_index,
        'offset_start': offset_start_index,
        'offset_end': offset_end_index,
        'background_start': background_start_index,
        'background_end': background_end_index,
    }

def calculate_delta_f_over_f(DA_signal, isosbestic_control, time_seconds, behavior_tags):
    # Option A: Per-trial baseline and per-trial regression
    use_per_trial = CONFIG['processing_options'].get('use_per_trial_baseline', False)

    sampling_rate = CONFIG['experimental_setup']['sampling_rate']
    kernel_size = CONFIG['signal_processing']['median_kernel_size']

    if use_per_trial:
        # Step 1: Calculate global offset using original method (same as legacy path)
        segments_DA_signal = detect_signal_segments(DA_signal, time_seconds, behavior_tags)
        segments_isosbestic_control = detect_signal_segments(isosbestic_control, time_seconds, behavior_tags)
        offset_intersection = min(segments_DA_signal['offset_end'], segments_isosbestic_control['offset_end']) - \
                                  max(segments_DA_signal['offset_start'], segments_isosbestic_control['offset_start'])
        if offset_intersection > CONFIG['analysis_parameters']['intersection_threshold']:
            segments_DA_signal['offset_start'] = max(segments_DA_signal['offset_start'], segments_isosbestic_control['offset_start'])
            segments_DA_signal['offset_end'] = min(segments_DA_signal['offset_end'], segments_isosbestic_control['offset_end'])
        segments_isosbestic_control['offset_start'] = segments_DA_signal['offset_start']
        segments_isosbestic_control['offset_end'] = segments_DA_signal['offset_end']

        # Step 2: Calculate global offset means (original method)
        offset_mean_DA_signal = np.mean(DA_signal[segments_DA_signal['offset_start']:segments_DA_signal['offset_end']])
        offset_mean_isosbestic_control = np.mean(isosbestic_control[segments_isosbestic_control['offset_start']:segments_isosbestic_control['offset_end']])

        # Step 3: Extract background region and remove offset (original method)
        bg_start, bg_end = segments_DA_signal['background_start'], segments_DA_signal['background_end']
        time_background = time_seconds[bg_start:bg_end]
        DA_signal_background = DA_signal[bg_start:bg_end] - offset_mean_DA_signal
        isosbestic_control_background = isosbestic_control[bg_start:bg_end] - offset_mean_isosbestic_control

        # Global processing (median, lowpass, detrend) applied on full background
        DA_signal_median_filtered = signal.medfilt(DA_signal_background, kernel_size=kernel_size)
        isosbestic_control_median_filtered = signal.medfilt(isosbestic_control_background, kernel_size=kernel_size)

        lowpass_b, lowpass_a = get_filter_coefficients(
            CONFIG['signal_processing']['lowpass_cutoff'], 2, 'low', sampling_rate
        )
        DA_signal_lowpass = signal.filtfilt(lowpass_b, lowpass_a, DA_signal_median_filtered)
        isosbestic_control_lowpass = signal.filtfilt(lowpass_b, lowpass_a, isosbestic_control_median_filtered)

        highpass_b, highpass_a = get_filter_coefficients(
            CONFIG['signal_processing']['baseline_cutoff'], 2, 'high', sampling_rate
        )
        DA_signal_detrended = signal.filtfilt(highpass_b, highpass_a, DA_signal_lowpass)
        isosbestic_control_detrended = signal.filtfilt(highpass_b, highpass_a, isosbestic_control_lowpass)

        # Prepare arrays to hold per-sample baseline and per-sample regression params
        DA_signal_baseline = np.full(time_background.shape, np.nan)
        isosbestic_control_baseline = np.full(time_background.shape, np.nan)
        correction_slope_arr = np.full(time_background.shape, 0.0)
        correction_intercept_arr = np.full(time_background.shape, 0.0)
        r_value_arr = np.full(time_background.shape, 0.0)

        # Parse trials and compute per-trial baseline and regression only
        trial_events = parse_behavior_events(behavior_tags)

        # baseline filter coeffs
        baseline_b, baseline_a = get_filter_coefficients(
            CONFIG['signal_processing']['baseline_cutoff'], 2, 'low', sampling_rate
        )

        # default global regression values (fallback)
        try:
            global_slope, global_intercept, global_r, _, _ = linregress_no_nan(isosbestic_control_detrended, DA_signal_detrended)
        except Exception:
            global_slope, global_intercept, global_r = 0.0, 0.0, 0.0

        # iterate trials and fill per-sample baseline and correction arrays
        for trial in trial_events:
            t_start = trial.get('t_time_start')
            t_end = trial.get('t_time_end')
            if t_start is None or t_end is None:
                continue
            mask = (time_background >= t_start) & (time_background <= t_end)

            # per-trial regression using detrended signals inside trial
            x = isosbestic_control_detrended[mask]
            y = DA_signal_detrended[mask]
            try:
                slope, intercept, r_val, _, _ = linregress_no_nan(x, y)
            except Exception:
                slope, intercept, r_val = global_slope, global_intercept, global_r

            correction_slope_arr[mask] = slope
            correction_intercept_arr[mask] = intercept
            r_value_arr[mask] = r_val

            # per-trial baseline: try to filter the lowpass signal within the trial

                # If the segment is too short for filtfilt padding, fallback to local mean
            seg = DA_signal_lowpass[mask]
            seg_base = signal.filtfilt(baseline_b, baseline_a, seg)
            DA_signal_baseline[mask] = seg_base

            seg = isosbestic_control_lowpass[mask]
            seg_base_iso = signal.filtfilt(baseline_b, baseline_a, seg)
            isosbestic_control_baseline[mask] = seg_base_iso

            # DA_signal_baseline[mask] = np.nanmean(DA_signal_lowpass)

        # Apply per-sample correction using per-trial slopes/intercepts
        DA_signal_delta_f = DA_signal_detrended - (correction_slope_arr * isosbestic_control_detrended + correction_intercept_arr)

        # Step 9: Calculate ΔF/F0 using per-trial baseline
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_f_over_f = DA_signal_delta_f / DA_signal_baseline

        # Step 10: Outlier detection and removal (global)
        outlier_segments, upper_threshold, lower_threshold = detect_outliers(delta_f_over_f)
        delta_f_over_f_before_outliers = delta_f_over_f.copy()
        for start_idx, end_idx in outlier_segments:
            delta_f_over_f[start_idx:end_idx] = np.nan
        final_delta_f_over_f = delta_f_over_f.copy()

        # Build outlier segments list (empty for per-trial path since handled globally)
        outlier_segments_global = outlier_segments

        # per-sample arrays (DA_signal_background, isosbestic_control_background, etc.)
        # were preallocated and filled per-trial above; use them directly for plotting and output.
        if not CONFIG['processing_options'].get('per_trial_regression', False):
            global_slope, global_intercept, global_r, _, _ = linregress_no_nan(
                isosbestic_control_detrended, DA_signal_detrended
            )
            # Apply per-sample correction
            DA_signal_delta_f = DA_signal_detrended - (global_slope * isosbestic_control_detrended + global_intercept)
            # Step 9: Calculate ΔF/F0
            delta_f_over_f = DA_signal_delta_f / DA_signal_baseline

            # Step 10: Outlier detection and removal
            outlier_segments, upper_threshold, lower_threshold = detect_outliers(delta_f_over_f)
            delta_f_over_f_before_outliers = delta_f_over_f.copy()
            for start_idx, end_idx in outlier_segments:
                delta_f_over_f[start_idx:end_idx] = np.nan
            final_delta_f_over_f = delta_f_over_f.copy()

        processing_data = {
            'time_background': time_background,
            'DA_signal_background': DA_signal_background,
            'isosbestic_control_background': isosbestic_control_background,

            'segments_DA_signal': segments_DA_signal,
            'segments_isosbestic_control': segments_isosbestic_control,

            'DA_signal_median_filtered': DA_signal_median_filtered,
            'isosbestic_control_median_filtered': isosbestic_control_median_filtered,

            'DA_signal_lowpass': DA_signal_lowpass,
            'isosbestic_control_lowpass': isosbestic_control_lowpass,

            'DA_signal_baseline': DA_signal_baseline,
            'isosbestic_control_baseline': isosbestic_control_baseline,

            'DA_signal_detrended': DA_signal_detrended,
            'isosbestic_control_detrended': isosbestic_control_detrended,

            'correction_slope': 0.0,
            'correction_intercept': 0.0,
            'r_value_correction': 0.0,
            'DA_signal_delta_f': DA_signal_delta_f,

            'delta_f_over_f_before_outliers': delta_f_over_f_before_outliers,
            'outlier_segments': outlier_segments_global,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,

            'final_delta_f_over_f': final_delta_f_over_f,
        }

        return processing_data

    # Default/legacy path: original global processing (unchanged)
    # Step 1: Segment detection
    segments_DA_signal = detect_signal_segments(DA_signal, time_seconds, behavior_tags)
    segments_isosbestic_control = detect_signal_segments(isosbestic_control, time_seconds, behavior_tags)
    offset_intersection = min(segments_DA_signal['offset_end'], segments_isosbestic_control['offset_end']) - \
                              max(segments_DA_signal['offset_start'], segments_isosbestic_control['offset_start'])
    if offset_intersection > CONFIG['analysis_parameters']['intersection_threshold']:
        segments_DA_signal['offset_start'] = max(segments_DA_signal['offset_start'], segments_isosbestic_control['offset_start'])
        segments_DA_signal['offset_end'] = min(segments_DA_signal['offset_end'], segments_isosbestic_control['offset_end'])
    segments_isosbestic_control['offset_start'] = segments_DA_signal['offset_start']
    segments_isosbestic_control['offset_end'] = segments_DA_signal['offset_end']

    # Step 2: Offset calculation
    offset_mean_DA_signal = np.mean(DA_signal[segments_DA_signal['offset_start']:segments_DA_signal['offset_end']])
    offset_mean_isosbestic_control = np.mean(isosbestic_control[segments_isosbestic_control['offset_start']:segments_isosbestic_control['offset_end']])

    # Step 3: Extract background regions and remove offset
    bg_start, bg_end = segments_DA_signal['background_start'], segments_DA_signal['background_end']
    time_background = time_seconds[bg_start:bg_end]
    DA_signal_background = DA_signal[bg_start:bg_end] - offset_mean_DA_signal
    isosbestic_control_background = isosbestic_control[bg_start:bg_end] - offset_mean_isosbestic_control

    # Step 4: Median filtering
    DA_signal_median_filtered = signal.medfilt(DA_signal_background, kernel_size=kernel_size)
    isosbestic_control_median_filtered = signal.medfilt(isosbestic_control_background, kernel_size=kernel_size)

    # Step 5: Low-pass filtering
    lowpass_b, lowpass_a = get_filter_coefficients(
        CONFIG['signal_processing']['lowpass_cutoff'],
        2,
        'low', sampling_rate
    )
    DA_signal_lowpass = signal.filtfilt(lowpass_b, lowpass_a, DA_signal_median_filtered)
    isosbestic_control_lowpass = signal.filtfilt(lowpass_b, lowpass_a, isosbestic_control_median_filtered)

    # Step 6: High-pass detrending
    highpass_b, highpass_a = get_filter_coefficients(
        CONFIG['signal_processing']['baseline_cutoff'], 2, 'high', sampling_rate
    )
    DA_signal_detrended = signal.filtfilt(highpass_b, highpass_a, DA_signal_lowpass)
    isosbestic_control_detrended = signal.filtfilt(highpass_b, highpass_a, isosbestic_control_lowpass)

    # Step 7: Extract baseline (low-frequency component) for normalization
    baseline_b, baseline_a = get_filter_coefficients(
        CONFIG['signal_processing']['baseline_cutoff'], 2, 'low', sampling_rate
    )
    DA_signal_baseline = signal.filtfilt(baseline_b, baseline_a, DA_signal_lowpass)
    isosbestic_control_baseline = signal.filtfilt(baseline_b, baseline_a, isosbestic_control_lowpass)

    # Step 8: Motion artifact correction using linear regression
    # Optionally perform per-trial regression (fit 410->470 within each trial)
    per_trial_reg = CONFIG['processing_options'].get('per_trial_regression', False)

    # Default: compute a global regression (used for plotting and as fallback)
    try:
        global_slope, global_intercept, global_r, _, _ = linregress_no_nan(
            isosbestic_control_detrended, DA_signal_detrended
        )
    except Exception:
        global_slope, global_intercept, global_r = 0.0, 0.0, 0.0

    if per_trial_reg:
        # Parse trial events to get trial boundaries
        trial_events = parse_behavior_events(behavior_tags)

        # Prepare arrays to hold per-sample slope/intercept/r
        correction_slope_arr = np.full_like(DA_signal_detrended, float(global_slope), dtype=float)
        correction_intercept_arr = np.full_like(DA_signal_detrended, float(global_intercept), dtype=float)
        r_value_arr = np.full_like(DA_signal_detrended, float(global_r), dtype=float)

        for trial in trial_events:
            t_start = trial.get('t_time_start')
            t_end = trial.get('t_time_end')
            if t_start is None or t_end is None:
                continue
            mask = (time_background >= t_start) & (time_background <= t_end)
            if np.sum(mask) < 3:
                continue
            x = isosbestic_control_detrended[mask]
            y = DA_signal_detrended[mask]
            try:
                slope, intercept, r_val, _, _ = linregress_no_nan(x, y)
            except Exception:
                slope, intercept, r_val = float(global_slope), float(global_intercept), float(global_r)
            correction_slope_arr[mask] = slope
            correction_intercept_arr[mask] = intercept
            r_value_arr[mask] = r_val

        # Apply per-sample correction
        DA_signal_delta_f = DA_signal_detrended - (correction_slope_arr * isosbestic_control_detrended + correction_intercept_arr)

        # Keep scalar global values for backward compatibility/plotting
        correction_slope = float(global_slope)
        correction_intercept = float(global_intercept)
        r_value = float(global_r)
    else:
        # Single global regression (original behavior)
        try:
            correction_slope, correction_intercept, r_value, _, _ = linregress(
                isosbestic_control_detrended, DA_signal_detrended
            )
        except Exception:
            correction_slope, correction_intercept, r_value = 0.0, 0.0, 0.0
        DA_signal_delta_f = DA_signal_detrended - (correction_slope * isosbestic_control_detrended + correction_intercept)

    # Step 9: Calculate ΔF/F0
    delta_f_over_f = DA_signal_delta_f / DA_signal_baseline

    # Step 10: Outlier detection and removal
    outlier_segments, upper_threshold, lower_threshold = detect_outliers(delta_f_over_f)
    delta_f_over_f_before_outliers = delta_f_over_f.copy()
    for start_idx, end_idx in outlier_segments:
        delta_f_over_f[start_idx:end_idx] = np.nan

    processing_data = {
        'time_background': time_background,
        'DA_signal_background': DA_signal_background,
        'isosbestic_control_background': isosbestic_control_background,

        'segments_DA_signal': segments_DA_signal,
        'segments_isosbestic_control': segments_isosbestic_control,

        'DA_signal_median_filtered': DA_signal_median_filtered,
        'isosbestic_control_median_filtered': isosbestic_control_median_filtered,

        'DA_signal_lowpass': DA_signal_lowpass,
        'isosbestic_control_lowpass': isosbestic_control_lowpass,

        'DA_signal_baseline': DA_signal_baseline,
        'isosbestic_control_baseline': isosbestic_control_baseline,

        'DA_signal_detrended': DA_signal_detrended,
        'isosbestic_control_detrended': isosbestic_control_detrended,

        'correction_slope': correction_slope,
        'correction_intercept': correction_intercept,
        'r_value_correction': r_value,
        'DA_signal_delta_f': DA_signal_delta_f,

        'delta_f_over_f_before_outliers': delta_f_over_f_before_outliers,
        'outlier_segments': outlier_segments,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,

        'final_delta_f_over_f': delta_f_over_f,
    }

    return processing_data


def detect_outliers(raw_data):
    """Detect outliers using rolling window statistics."""
    sampling_rate = CONFIG['experimental_setup']['sampling_rate']
    window_size = int(sampling_rate * 60 * CONFIG['signal_processing']['outlier_window_minutes'])
    # window_size = len(raw_data) // 1

    threshold_factor = CONFIG['signal_processing']['outlier_threshold_factor']
    expand_seconds = CONFIG['signal_processing']['outlier_expand_seconds']
    expand_points = int(sampling_rate * expand_seconds)
    gap_tolerance = int(sampling_rate * CONFIG['signal_processing']['outlier_gap_seconds'])

    series = pd.Series(raw_data)
    rolling_stats = series.rolling(window=window_size, center=True, min_periods=1)
    local_mean = rolling_stats.mean().values
    local_std = rolling_stats.std().values

    upper_threshold = local_mean + threshold_factor * local_std
    lower_threshold = local_mean - threshold_factor * local_std

    outlier_mask = (raw_data > upper_threshold) | (raw_data < lower_threshold)
    outlier_indices = np.where(outlier_mask)[0]

    if len(outlier_indices) == 0:
        return [], upper_threshold, lower_threshold

    # Group consecutive outliers
    segments = []
    start = outlier_indices[0]
    prev = outlier_indices[0]

    for idx in outlier_indices[1:]:
        if idx - prev <= gap_tolerance:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx

    segments.append((start, prev))

    expanded_segments = []
    for (s, e) in segments:
        s_expanded = max(0, s - expand_points)
        e_expanded = min(len(raw_data) - 1, e + expand_points)
        expanded_segments.append((s_expanded, e_expanded))

    return expanded_segments, upper_threshold, lower_threshold


def parse_behavior_events(behavior_tags):
    """Parse behavior events into trial structure."""
    behavior_data = behavior_tags[behavior_tags['Behavior'].isin(['t', 'w', 'g', 'c', 'v', 'i', 'q'])].copy()
    behavior_data = behavior_data.sort_values('Start (s)').reset_index(drop=True)

    trial_events = []
    current_trial = None
    trial_counter = 1

    for _, row in behavior_data.iterrows():
        behavior = row['Behavior']
        start_time = row['Start (s)']
        if behavior == 't':
            if current_trial is not None:
                current_trial['t_time_end'] = start_time
                trial_events.append(current_trial)
                trial_counter += 1
            current_trial = {
                'trial_id': trial_counter,
                't_time_start': start_time,
                'current_choice': None,
                'i_time': None,
                'o_time': None,
                'dm_time': None,
            }
        elif behavior in ['w', 'g'] and current_trial is not None:
            current_trial.update({
                'i_time': start_time,
                'current_choice': behavior
            })
        elif behavior in ['c', 'v'] and current_trial is not None:
            current_trial['o_time'] = start_time
        elif behavior in ['i', 'q'] and current_trial is not None:
            current_trial['dm_time'] = start_time

    return trial_events


def create_analysis_plots(mouse_id, phase_id, brain_region, DA_signal, isosbestic_control, time_seconds, processing_data, trial_events):

    plots_directory = os.path.join(CONFIG['data_paths']['plots_directory'], brain_region)
    time_background = processing_data['time_background']
    segments_DA_signal = processing_data['segments_DA_signal']
    segments_isosbestic_control = processing_data['segments_isosbestic_control']

    # Create main figure with subplots
    fig = plt.figure(figsize=(28, 16))
    gs_main = fig.add_gridspec(3, 3, wspace=0.25, hspace=0.4)

    # Plot 1: Signal Segmentation
    gs_seg = gs_main[0, 0].subgridspec(2, 1, hspace=0.1)
    ax_seg1 = fig.add_subplot(gs_seg[0, 0])
    ax_seg2 = fig.add_subplot(gs_seg[1, 0], sharex=ax_seg1)

    ax_seg1.plot(time_seconds, DA_signal, 'b-', linewidth=0.8, alpha=0.7, label='Raw DA Signal')
    ax_seg1.axvspan(time_seconds[segments_DA_signal['offset_start']], time_seconds[segments_DA_signal['offset_end']],
                    alpha=0.25, color='red', label='Offset Region')
    ax_seg1.axvspan(time_seconds[segments_DA_signal['background_start']], time_seconds[segments_DA_signal['background_end']],
                    alpha=0.25, color='green', label='Background Region')
    ax_seg1.set_title('1. Signal Segmentation')
    ax_seg1.set_ylabel('Fluorescence (AU)')
    ax_seg1.legend()
    plt.setp(ax_seg1.get_xticklabels(), visible=False)

    ax_seg2.plot(time_seconds, isosbestic_control, 'purple', linewidth=0.8, alpha=0.7, label='Raw Isosbestic Control')
    ax_seg2.axvspan(time_seconds[segments_isosbestic_control['offset_start']], time_seconds[segments_isosbestic_control['offset_end']],
                    alpha=0.25, color='red')
    ax_seg2.axvspan(time_seconds[segments_isosbestic_control['background_start']], time_seconds[segments_isosbestic_control['background_end']],
                    alpha=0.25, color='green')
    ax_seg2.set_xlabel('Time (s)')
    ax_seg2.set_ylabel('Fluorescence (AU)')
    ax_seg2.legend()

    # Plot 2: Filtering Steps
    ax_filt = fig.add_subplot(gs_main[0, 1])
    ax_filt.plot(time_background, processing_data['DA_signal_background'], 'b-',
                 linewidth=0.8, alpha=0.7, label='After Offset Removal')
    ax_filt.plot(time_background, processing_data['DA_signal_median_filtered'], 'g-',
                 linewidth=0.8, label='After Median Filter')
    ax_filt.plot(time_background, processing_data['DA_signal_lowpass'], 'r-',
                 linewidth=1, label='After Lowpass Filter')
    # Plot DA (470nm) baseline curve on the same plot (use a dashed dark green line)
    ax_filt.plot(time_background, processing_data['DA_signal_baseline'], color='darkgreen',
                 linestyle='--', linewidth=1, alpha=0.9, label='DA Baseline (F)')
    # Highlight trial regions similar to Plot 7 (use semi-transparent colored spans and labels)
    choice_colors = {'w': 'red', 'g': 'blue'}
    for ti, trial in enumerate(trial_events):
        if trial.get('t_time_start') is None:
            continue
        trial_start = trial['t_time_start']
        trial_end = trial.get('t_time_end', trial_start + 10)
        color = choice_colors.get(trial.get('current_choice'), 'grey')
        ax_filt.axvspan(trial_start, trial_end, color=color, alpha=0.12,
                        label=f"Choice {trial['current_choice'].upper()}" if ti == 0 and trial.get('current_choice') in choice_colors else "")
        # Add trial id label near the top of the axis
        ylim = ax_filt.get_ylim()
        y_text = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        ax_filt.text(trial_start, y_text, f"T{trial['trial_id']}", ha='left', va='top', fontsize=7, fontweight='bold')
    ax_filt.set_title('2. Filtering Pipeline (470nm)')
    ax_filt.set_xlabel('Time (s)')
    ax_filt.set_ylabel('Fluorescence (AU)')
    ax_filt.legend()

    # Plot 3: Baseline Extraction
    ax_base = fig.add_subplot(gs_main[0, 2])

    color1 = 'tab:blue'
    ax_base.plot(time_background, processing_data['isosbestic_control_baseline'], color=color1,
                 linewidth=1.0, label='isosbestic Baseline (410)')
    # ax_base.plot(time_background, processing_data['DA_signal_baseline'], color='tab:red',
    #              linewidth=1.0, label='DA Baseline (F)')
    ax_base.set_ylabel('iso baseline', color=color1)
    ax_base.tick_params(axis='y', labelcolor=color1)
    m = np.nanmean(processing_data['isosbestic_control_baseline'])
    ax_base.set_ylim(m - m*0.05, m + m*0.05)

    ax2 = ax_base.twinx()
    color2 = 'tab:green'
    ax2.plot(time_background, processing_data['DA_signal_baseline'], color=color2,
             linewidth=1.0, label='DA Baseline (F)')
    ax2.set_ylabel('DA Baseline (F)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax_base.set_title('3. Baseline Extraction')
    lines_1, labels_1 = ax_base.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax_base.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    ax_base.set_xlabel('Time (s)')
    m = np.nanmean(processing_data['DA_signal_baseline'])
    ax2.set_ylim(m - m*0.05, m + m*0.05)

    # Plot 4: Motion Artifact Correction
    gs_corr = gs_main[1, 0].subgridspec(1, 2, wspace=0.3)
    ax_corr_scatter = fig.add_subplot(gs_corr[0, 0])
    ax_corr_time = fig.add_subplot(gs_corr[0, 1])

    # Scatter plot for correlation
    x_isosbestic = processing_data['isosbestic_control_detrended']
    y_DA_signal_before = processing_data['DA_signal_detrended']
    y_DA_signal_after = processing_data['DA_signal_delta_f']
    sample_size = CONFIG['analysis_parameters']['scatter_sample_size']

    indices = np.random.choice(len(x_isosbestic), size=sample_size, replace=False)
    ax_corr_scatter.scatter(x_isosbestic[indices], y_DA_signal_before[indices],
                            alpha=0.5, s=1.5, color='blue', label='Data Points')

    fit_x = np.linspace(x_isosbestic.min(), x_isosbestic.max(), 100)
    fit_y = processing_data['correction_slope'] * fit_x + processing_data['correction_intercept']
    ax_corr_scatter.plot(fit_x, fit_y, 'r-', linewidth=2,
                         label=f'Fit: R²={processing_data["r_value_correction"] ** 2:.3f}')
    ax_corr_scatter.set_title('4a. Motion Correlation')
    ax_corr_scatter.set_xlabel('detrended isosbestic')
    ax_corr_scatter.set_ylabel('detrended DA signal')
    ax_corr_scatter.legend()

    # Time series before/after correction
    ax_corr_time.plot(time_background, y_DA_signal_before, 'b-', linewidth=0.8,
                      alpha=0.7, label='Before Correction')
    ax_corr_time.plot(time_background, y_DA_signal_after, 'r-', linewidth=0.8,
                      label='After Correction')
    ax_corr_time.set_title('4b. Motion Correction')
    ax_corr_time.set_xlabel('Time (s)')
    ax_corr_time.set_ylabel('ΔF')
    ax_corr_time.legend()

    # Plot 5: ΔF/F Calculation
    ax_df_f = fig.add_subplot(gs_main[1, 1])
    ax_df_f.plot(time_background, processing_data['DA_signal_delta_f'], 'b-',
                 linewidth=0.8, alpha=0.7, label='ΔF')
    ax_df_f.plot(time_background, processing_data['DA_signal_baseline'], 'g-',
                 linewidth=1, alpha=0.8, label='F')
    ax_df_f_twin = ax_df_f.twinx()
    ax_df_f_twin.plot(time_background, processing_data['delta_f_over_f_before_outliers'], 'r-',
                      linewidth=1, label='ΔF/F0')
    ax_df_f.set_title('5. ΔF/F0 Calculation')
    ax_df_f.set_xlabel('Time (s)')
    ax_df_f.set_ylabel('ΔF, F')
    ax_df_f_twin.set_ylabel('ΔF/F')
    ax_df_f.legend(loc='upper left')
    ax_df_f_twin.legend(loc='upper right')

    # Plot 6: Outlier Detection and Removal
    ax_outlier = fig.add_subplot(gs_main[1, 2])
    ax_outlier.plot(time_background, processing_data['delta_f_over_f_before_outliers'],
                    'b-', linewidth=0.8, alpha=0.7, label='Before Outlier Removal')
    ax_outlier.plot(time_background, processing_data['final_delta_f_over_f'],
                    'r-', linewidth=0.8, label='After Outlier Removal')
    ax_outlier.plot(time_background, processing_data['upper_threshold'], color='grey', linestyle='--',
                       linewidth=1, label='Upper Threshold')
    ax_outlier.plot(time_background, processing_data['lower_threshold'], color='grey', linestyle='--',
                          linewidth=1, label='Lower Threshold')

    # Highlight outlier regions
    for start_idx, end_idx in processing_data['outlier_segments']:
        ax_outlier.axvspan(time_background[start_idx], time_background[end_idx],
                           color='grey', alpha=0.3,
                           label='Outliers' if start_idx == processing_data['outlier_segments'][0][0] else "")
    ax_outlier.set_title('6. Outlier Detection & Removal')
    ax_outlier.set_xlabel('Time (s)')
    ax_outlier.set_ylabel('ΔF/F')
    ax_outlier.legend()

    # Plot 7: Final Signal with Behavior Events
    ax_final = fig.add_subplot(gs_main[2, :])
    ax_final.plot(time_background, processing_data['final_delta_f_over_f'], 'k-',
                  linewidth=0.8, alpha=0.9, label='Final ΔF/F')
    # Add behavior events
    choice_colors = {'w': 'red', 'g': 'blue'}
    for trial in trial_events:
        if trial.get('current_choice') in choice_colors:
            trial_start = trial['t_time_start']
            trial_end = trial.get('t_time_end', trial_start + 10)
            color = choice_colors[trial['current_choice']]

            ax_final.axvspan(trial_start, trial_end, color=color, alpha=0.15,
                             label=f"Choice {trial['current_choice'].upper()}" if trial == trial_events[0] else "")

            # Add trial markers
            ax_final.text(trial_start, ax_final.get_ylim()[1] * 0.95, f"T{trial['trial_id']}",
                          ha='left', va='top', fontsize=8, fontweight='bold')

            # Add event markers
            if 'i_time' in trial and trial['i_time'] is not None:
                ax_final.axvline(x=trial['i_time'], color='green', linestyle='--',
                                 linewidth=1, alpha=0.7, label='Input' if trial == trial_events[0] else "")
            if 'o_time' in trial and trial['o_time'] is not None:
                ax_final.axvline(x=trial['o_time'], color='orange', linestyle='--',
                                 linewidth=1, alpha=0.7, label='Output' if trial == trial_events[0] else "")

    ax_final.set_title('7. Final ΔF/F0 Signal with Behavioral Events')
    ax_final.set_xlabel('Time (s)')
    ax_final.set_ylabel('ΔF/F')
    ax_final.grid(True, alpha=0.3)

    # Clean up legend
    handles, labels = ax_final.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_final.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Add main title
    fig.suptitle(f'{mouse_id} - {phase_id} - {brain_region} Signal Processing Pipeline',
                 fontsize=20, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plots_directory = plots_directory.replace('\\', '/')
    os.makedirs(plots_directory, exist_ok=True)
    plt.savefig(
        os.path.join(plots_directory, f'{mouse_id}_{phase_id}_{brain_region}_processing_pipeline.png'),
        dpi=CONFIG['analysis_parameters']['plot_resolution'], bbox_inches='tight'
    )
    plt.close(fig)



def create_continuous_entry(mouse_id, phase_id, brain_region, time_background, df_f_signal, trial_events):

    event_times = {
        'trial_boundaries': [],  # t_time_start 和 t_time_end
        'i_time': [],  # i_time (进管时间)
        'o_time': [],  # o_time (出管时间)
        'decision_times': [],  # dm_time (决策时间)
        'trial_info': []  # 每个trial的详细信息
    }

    for trial in trial_events:
        # 收集trial边界时间
        event_times['trial_boundaries'].extend([
            trial['t_time_start'],
            trial['t_time_end']
        ])

        # 收集各种事件时间
        if trial.get('i_time') is not None:
            event_times['i_time'].append(trial['i_time'])
        if trial.get('o_time') is not None:
            event_times['o_time'].append(trial['o_time'])
        if trial.get('dm_time') is not None:
            event_times['decision_times'].append(trial['dm_time'])

        # 保存trial详细信息
        event_times['trial_info'].append({
            'trial_id': trial['trial_id'],
            't_start': trial['t_time_start'],
            't_end': trial.get('t_time_end'),
            'i_time': trial.get('i_time'),
            'o_time': trial.get('o_time'),
            'dm_time': trial.get('dm_time'),
            'choice': trial.get('current_choice')
        })

    for key in ['trial_boundaries', 'i_time', 'o_time', 'decision_times']:
        event_times[key] = sorted(list(set([t for t in event_times[key] if t is not None])))

    continuous_entry = {
        'mouse_id': mouse_id,
        'phase_id': phase_id,
        'brain_region': brain_region,
        'time_array': time_background,  # 时间序列
        'df_f_continuous': df_f_signal,  # 连续的 ΔF/F 信号（包含 NaN）
        'event_times': event_times,  # 所有事件时间点
    }

    return continuous_entry

def process_single_experiment(mouse_id, phase_id, channel_prefix):

    data_directory = os.path.join(CONFIG['data_paths']['input_directory'], mouse_id, phase_id)
    behavior_file = os.path.join(data_directory, 'top.csv')
    fluorescence_file = os.path.join(data_directory, 'fluorescence.csv')
    behavior_tags = pd.read_csv(behavior_file)

    with open(fluorescence_file, "r") as f:
        first_line = f.readline().strip()
    fluorescence_data = pd.read_csv(fluorescence_file, skiprows = 0 if "TimeStamp" in first_line else 1)
    fluorescence_data = fluorescence_data.apply(pd.to_numeric, errors='coerce')

    channels = CONFIG['experimental_setup']['channels']

    DA_signal = fluorescence_data[f"{channel_prefix}-{channels[1]}"].values
    isosbestic_control = fluorescence_data[f"{channel_prefix}-{channels[0]}"].values
    time_seconds = fluorescence_data["TimeStamp"].values / 1000

    min_length = min(len(DA_signal), len(isosbestic_control), len(time_seconds))
    DA_signal = DA_signal[:min_length]
    isosbestic_control = isosbestic_control[:min_length]
    time_seconds = time_seconds[:min_length]

    processing_data = calculate_delta_f_over_f(
        DA_signal, isosbestic_control, time_seconds, behavior_tags
    )

    # Parse behavior events
    trial_events = parse_behavior_events(behavior_tags)
    brain_region = CONFIG['experimental_setup']['brain_regions'][channel_prefix]
    time_background = processing_data['time_background']
    final_delta_f_over_f = processing_data['final_delta_f_over_f']

    # Generate plots if enabled
    if CONFIG['processing_options']['generate_plots']:
        create_analysis_plots(mouse_id, phase_id, brain_region, DA_signal, isosbestic_control, time_seconds, processing_data, trial_events)

    # Create dataset entries
    dataset_entries = []

    for trial_index, trial in enumerate(trial_events):

        trial_start_time = trial['t_time_start']
        trial_end_time = trial['t_time_end']
        time_mask = (time_background >= trial_start_time) & (time_background <= trial_end_time)
        trial_delta_f_over_f = final_delta_f_over_f[time_mask]

        # Get next choice for analysis
        next_choice = None
        if trial_index + 1 < len(trial_events):
            next_choice = trial_events[trial_index + 1].get('current_choice')

        prev_choice = None
        if trial_index - 1 >= 0:
            prev_choice = trial_events[trial_index - 1].get('current_choice')

        dataset_entries.append({
            'mouse_id': mouse_id,
            'phase_id': phase_id,
            'brain_region': brain_region,
            'trial_id': trial['trial_id'],
            'df_f': trial_delta_f_over_f,
            'i_time': trial['i_time'],
            'o_time': trial['o_time'],
            'dm_time': trial['dm_time'],
            't_time_start': trial['t_time_start'],
            't_time_end': trial['t_time_end'],
            'current_choice': trial['current_choice'],
            'next_choice': next_choice,
            'prev_choice': prev_choice,
        })

    continuous_entry = create_continuous_entry(
        mouse_id, phase_id, brain_region,
        time_background, final_delta_f_over_f,
        trial_events
    )

    return dataset_entries, continuous_entry


def process_experiment_wrapper(args):
    """Wrapper function for multiprocessing."""
    return process_single_experiment(*args)


def get_tasks(mice_to_process):
    target_phases = CONFIG['processing_options']['target_phases']
    target_channels = CONFIG['processing_options']['target_channels']
    processing_tasks = []
    for mouse_id in mice_to_process:
        for phase_id in target_phases:
            for channel_prefix in target_channels:
                processing_tasks.append((mouse_id, phase_id, channel_prefix))
    return processing_tasks

def process_all_tasks(processing_tasks):
    all_dataset_entries = []
    all_continuous_entries = []
    if CONFIG['processing_options']['enable_multiprocessing']:
        max_workers = CONFIG['processing_options']['max_workers'] or min(cpu_count() // 2, len(processing_tasks))
        with Pool(max_workers) as pool:
            results = list(tqdm(
                pool.imap(process_experiment_wrapper, processing_tasks),
                total=len(processing_tasks),
                desc='Processing Tasks',
                unit='task'
            ))
        for trial_entries, continuous_entry in results:
            all_dataset_entries.extend(trial_entries)
            all_continuous_entries.append(continuous_entry)
    else:
        with tqdm(total=len(processing_tasks), desc='Processing Tasks', unit='task') as progress_bar:
            for mouse_id, phase_id, channel_prefix in processing_tasks:
                progress_bar.set_postfix_str(f'{mouse_id}_{phase_id}_{channel_prefix}')
                trial_entries, continuous_entry = process_single_experiment(mouse_id, phase_id, channel_prefix)
                all_dataset_entries.extend(trial_entries)
                all_continuous_entries.append(continuous_entry)
                progress_bar.update(1)

    return all_dataset_entries, all_continuous_entries  # 修改返回值

def calculate_time_intervals(df):
    # 将时间列转换为数值类型
    time_cols = ['i_time', 'o_time', 'dm_time', 't_time_start', 't_time_end']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算时间间隔
    df['trial_start_to_dm'] = df['dm_time'] - df['t_time_start']
    df['dm_to_in'] = df['i_time'] - df['dm_time']
    df['trial_start_to_in'] = df['i_time'] - df['t_time_start']
    df['in_to_out'] = df['o_time'] - df['i_time']
    df['out_to_trial_end'] = df['t_time_end'] - df['o_time']

    # 排序，保证 trial 对齐
    df = df.sort_values(['mouse_id', 'phase_id', 'brain_region', 'trial_id'])

    # 出管到下一个 trial 进管
    df['next_trial_i'] = df.groupby(['mouse_id', 'phase_id', 'brain_region'])['i_time'].shift(-1)
    df['out_to_next_trial_i'] = df['next_trial_i'] - df['o_time']

    # 上一次出管到当前 trial 进管
    df['prev_trial_o'] = df.groupby(['mouse_id', 'phase_id', 'brain_region', ])['o_time'].shift(1)
    df['prev_out_to_current_in'] = df['i_time'] - df['prev_trial_o']
    # 上一次出管到当前 trial 决策
    df['prev_out_to_current_dm'] = df['dm_time'] - df['prev_trial_o']

    return df


def save_continuous_pickle(all_continuous_entries):
    """保存连续信号数据到pickle文件"""
    # Build path with pathlib to avoid mixed separators on Windows and ensure
    # the directory exists before attempting to write the file.
    datasets_dir = Path(CONFIG['data_paths']['datasets_directory'])
    continuous_pickle_path = datasets_dir / CONFIG['data_paths']['continuous_dataset_name']

    # Ensure parent directory exists
    continuous_pickle_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for analysis and write pickle
    continuous_df = pd.DataFrame(all_continuous_entries)
    with open(continuous_pickle_path, 'wb') as f:
        pickle.dump(continuous_df, f)


def save_pickle(all_dataset_entries):
    datasets_dir = Path(CONFIG['data_paths']['datasets_directory'])
    pickle_output_path = datasets_dir / CONFIG['data_paths']['dataset_name']

    # Ensure parent directory exists
    pickle_output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dataset_entries = pd.DataFrame(all_dataset_entries)
    all_dataset_entries = calculate_time_intervals(all_dataset_entries)
    with open(pickle_output_path, 'wb') as pickle_file:
        pickle.dump(all_dataset_entries, pickle_file)

def main():
    mice_to_process = get_mice_ids()
    processing_tasks = get_tasks(mice_to_process)
    all_dataset_entries, all_continuous_entries = process_all_tasks(processing_tasks)  # 修改
    save_pickle(all_dataset_entries)
    save_continuous_pickle(all_continuous_entries)  # 新增

if __name__ == "__main__":
    main()