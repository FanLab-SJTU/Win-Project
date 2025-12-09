import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')

# 全局配置
align_point = 'i_dm'  # 'i' or 'dm' or 'i_dm' or 'cross_trial'

CONFIG = {
    'data_type': 'male_high_rank',
    'sampling_rate': 50,  # Hz

    # Single-trial windows
    'trial_windows': {
        'before_dm': 2,
        'dm_to_in': 1,
        'before_in': 3,
        'middle': 6.0,
        'after_out': 3,
        'out_to_next_in': 33,
        'prev_out_to_curr_in': 33
    },

    # Cross-trial windows
    'cross_trial_windows': {
        'before_dm_1': 2,
        'dm_to_in_1': 1,
        'in_to_out_1': 6,
        'after_out_1': 3,
        'out1_to_dm2': 24,
        'before_dm_2': 2,
        'dm_to_in_2': 1,
        'in_to_out_2': 6,
        'after_out_2': 3,
    },

    'phase_order': ['baseline', 'win-1', 'win-2', 'win-3', 'win-4'],
    'brain_regions': ['mPFC','NAc'],
    'normalization': {
        'lower_q': 5,
        'upper_q': 95
    },
    'max_trials': [20, 30, 30, 30, 30]
}


def get_file_paths(config, align_point):
    """生成所有需要的文件路径"""
    data_type = config['data_type']

    if align_point == 'cross_trial':
        return {
            'input_data': f'data/reorganized_dataset/{data_type}/DA_datasets.pkl',
            'warped_output': f'data/reorganized_dataset/{data_type}/warped_DA_datasets_{align_point}.pkl',
        }
    else:
        return {
            'input_data': f'data/reorganized_dataset/{data_type}/DA_datasets.pkl',
            'warped_output': f'data/reorganized_dataset/{data_type}/warped_DA_datasets_{align_point}.pkl',
            'hmm_results': f'./data/results/HMM/{data_type}/detailed_posteriors.csv',
            'matrix_output_template': f'./data/reorganized_dataset/{data_type}/{{}}_matrices_{align_point}.pkl'
        }


def linear_warp(segment, target_len, sampling_rate):
    """对信号段进行线性时间规整"""
    if len(segment) < 2:
        return np.full(int(target_len * sampling_rate), np.nan)

    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, int(target_len * sampling_rate))
    f = interp1d(x_old, segment, kind='linear', fill_value='extrapolate')
    return f(x_new)


def create_warped_time(trial_data, next_trial_data, prev_trial_data, config):
    """
    创建单试验的warped信号（支持i, dm, i_dm对齐模式）

    Returns:
        (main_warped_signal, warped_out_to_next_in_signal, warped_prev_out_to_curr_in_signal)
    """
    sampling_rate = config['sampling_rate']
    windows = config['trial_windows']

    dm_time = trial_data['dm_time']
    i_time = trial_data['i_time']
    o_time = trial_data['o_time']
    t_start = trial_data['t_time_start']
    t_end = trial_data['t_time_end']
    df_f = np.array(trial_data['df_f'])

    if not np.isnan(dm_time):
        dm_time_idx = int((dm_time - t_start) * sampling_rate)
    i_time_idx = int((i_time - t_start) * sampling_rate)
    o_time_idx = int((o_time - t_start) * sampling_rate)

    if align_point == 'i_dm':
        start_of_interest = dm_time - windows['before_dm']
        end_of_interest = o_time + windows['after_out']

        original_timeline = np.linspace(t_start, t_end, len(df_f))
        window_indices = (original_timeline >= start_of_interest) & (original_timeline <= end_of_interest)

        start_idx = int((max(t_start, start_of_interest) - t_start) * sampling_rate)
        end_idx = int((min(t_end, end_of_interest) - t_start) * sampling_rate)

        before_dm_seg = df_f[start_idx:dm_time_idx]
        dm_to_in_seg = df_f[dm_time_idx:i_time_idx]
        in_to_out_seg = df_f[i_time_idx:o_time_idx]
        after_out_seg = df_f[o_time_idx:end_idx]

        if np.any(np.isnan(df_f[window_indices])) or len(before_dm_seg) <= 20 or len(dm_to_in_seg) <= 10 \
                or len(in_to_out_seg) <= 60 or len(after_out_seg) <= 30:
            return None, None, None

        warped_before_dm = linear_warp(before_dm_seg, windows['before_dm'], sampling_rate)
        warped_dm_to_in = linear_warp(dm_to_in_seg, windows['dm_to_in'], sampling_rate)
        warped_in_to_out = linear_warp(in_to_out_seg, windows['middle'], sampling_rate)
        warped_after_out = linear_warp(after_out_seg, windows['after_out'], sampling_rate)

        main_warped_signal = np.concatenate([warped_before_dm, warped_dm_to_in,
                                             warped_in_to_out, warped_after_out])

        warped_out_to_next_in = None
        if next_trial_data is not None:
            out_to_next_in_start = o_time + windows['after_out']
            next_i_time = next_trial_data['i_time']

            combined_data = np.concatenate([df_f, next_trial_data['df_f']])
            combined_start_time = t_start

            out_to_next_in_start_idx = int((out_to_next_in_start - combined_start_time) * sampling_rate)
            out_to_next_in_end_idx = int((next_i_time - combined_start_time) * sampling_rate)

            out_to_next_in_seg = combined_data[out_to_next_in_start_idx:out_to_next_in_end_idx]

            if not np.any(np.isnan(out_to_next_in_seg)):
                warped_out_to_next_in = linear_warp(
                    out_to_next_in_seg,
                    windows['out_to_next_in'],
                    sampling_rate
                )

        warped_prev_out_to_curr_in = None
        if prev_trial_data is not None:
            prev_o_time = prev_trial_data['o_time']
            prev_out_to_curr_in_start = prev_o_time + windows['after_out']
            prev_out_to_curr_in_end = i_time

            combined_data = np.concatenate([prev_trial_data['df_f'], df_f])
            combined_start_time = prev_trial_data['t_time_start']

            prev_out_to_curr_in_start_idx = int((prev_out_to_curr_in_start - combined_start_time) * sampling_rate)
            prev_out_to_curr_in_end_idx = int((prev_out_to_curr_in_end - combined_start_time) * sampling_rate)
            prev_out_to_curr_in_seg = combined_data[prev_out_to_curr_in_start_idx:prev_out_to_curr_in_end_idx]

            if not np.any(np.isnan(prev_out_to_curr_in_seg)):
                warped_prev_out_to_curr_in = linear_warp(
                    prev_out_to_curr_in_seg,
                    windows['prev_out_to_curr_in'],
                    sampling_rate
                )

        return main_warped_signal, warped_out_to_next_in, warped_prev_out_to_curr_in

    elif align_point == 'i':
        start_of_interest = i_time - windows['before_in']
        end_of_interest = o_time + windows['after_out']

        original_timeline = np.linspace(t_start, t_end, len(df_f))
        window_indices = (original_timeline >= start_of_interest) & (original_timeline <= end_of_interest)

        start_idx = int((max(t_start, start_of_interest) - t_start) * sampling_rate)
        before_seg = df_f[start_idx:i_time_idx]

        if np.any(np.isnan(df_f[window_indices])) or len(before_seg) <= 30:
            return None, None, None

        warped_before = linear_warp(before_seg, windows['before_in'], sampling_rate)
        middle_seg = df_f[i_time_idx:o_time_idx]
        warped_middle = linear_warp(middle_seg, windows['middle'], sampling_rate)
        end_idx = int((min(t_end, end_of_interest) - t_start) * sampling_rate)
        after_seg = df_f[o_time_idx:end_idx]
        warped_after = linear_warp(after_seg, windows['after_out'], sampling_rate)

        return np.concatenate([warped_before, warped_middle, warped_after]), None, None

    elif align_point == 'dm':
        start_of_interest = dm_time - windows['before_in']
        end_of_interest = o_time + windows['after_out']

        original_timeline = np.linspace(t_start, t_end, len(df_f))
        window_indices = (original_timeline >= start_of_interest) & (original_timeline <= end_of_interest)

        start_idx = int((max(t_start, start_of_interest) - t_start) * sampling_rate)
        before_seg = df_f[start_idx:dm_time_idx]

        if np.any(np.isnan(df_f[window_indices])) or len(before_seg) <= 50:
            return None, None, None

        warped_before = linear_warp(before_seg, windows['before_in'], sampling_rate)
        middle_seg = df_f[dm_time_idx:o_time_idx]
        warped_middle = linear_warp(middle_seg, windows['middle'], sampling_rate)
        end_idx = int((min(t_end, end_of_interest) - t_start) * sampling_rate)
        after_seg = df_f[o_time_idx:end_idx]
        warped_after = linear_warp(after_seg, windows['after_out'], sampling_rate)

        return np.concatenate([warped_before, warped_middle, warped_after]), None, None

    else:
        return None, None, None


def create_cross_trial_warped_signal(trial_1, trial_2, config):
    """
    创建跨越两个连续trial的warped信号

    Returns:
        (warped_signal, segment_indices)
    """
    sampling_rate = config['sampling_rate']
    windows = config['cross_trial_windows']

    dm_time_1 = trial_1['dm_time']
    i_time_1 = trial_1['i_time']
    o_time_1 = trial_1['o_time']
    t_start_1 = trial_1['t_time_start']
    df_f_1 = np.array(trial_1['df_f'])

    dm_time_2 = trial_2['dm_time']
    i_time_2 = trial_2['i_time']
    o_time_2 = trial_2['o_time']
    t_start_2 = trial_2['t_time_start']
    df_f_2 = np.array(trial_2['df_f'])

    dm_time_idx_1 = int((dm_time_1 - t_start_1) * sampling_rate)
    i_time_idx_1 = int((i_time_1 - t_start_1) * sampling_rate)
    o_time_idx_1 = int((o_time_1 - t_start_1) * sampling_rate)

    dm_time_idx_2 = int((dm_time_2 - t_start_2) * sampling_rate)
    i_time_idx_2 = int((i_time_2 - t_start_2) * sampling_rate)
    o_time_idx_2 = int((o_time_2 - t_start_2) * sampling_rate)

    before_dm_start_idx_1 = max(0, dm_time_idx_1 - int(windows['before_dm_1'] * sampling_rate))
    seg_before_dm_1 = df_f_1[before_dm_start_idx_1:dm_time_idx_1]

    seg_dm_to_in_1 = df_f_1[dm_time_idx_1:i_time_idx_1]
    seg_in_to_out_1 = df_f_1[i_time_idx_1:o_time_idx_1]

    after_out_end_idx_1 = min(len(df_f_1), o_time_idx_1 + int(windows['after_out_1'] * sampling_rate))
    seg_after_out_1 = df_f_1[o_time_idx_1:after_out_end_idx_1]

    out_plus_3s_idx_1 = after_out_end_idx_1
    before_dm_start_idx_2 = max(0, dm_time_idx_2 - int(windows['before_dm_2'] * sampling_rate))

    seg_out1_to_dm2 = np.concatenate([
        df_f_1[out_plus_3s_idx_1:],
        df_f_2[:before_dm_start_idx_2]
    ])

    seg_before_dm_2 = df_f_2[before_dm_start_idx_2:dm_time_idx_2]
    seg_dm_to_in_2 = df_f_2[dm_time_idx_2:i_time_idx_2]
    seg_in_to_out_2 = df_f_2[i_time_idx_2:o_time_idx_2]

    after_out_end_idx_2 = min(len(df_f_2), o_time_idx_2 + int(windows['after_out_2'] * sampling_rate))
    seg_after_out_2 = df_f_2[o_time_idx_2:after_out_end_idx_2]

    segments = [
        seg_before_dm_1, seg_dm_to_in_1, seg_in_to_out_1, seg_after_out_1,
        seg_out1_to_dm2,
        seg_before_dm_2, seg_dm_to_in_2, seg_in_to_out_2, seg_after_out_2
    ]

    warped_segments = []
    segment_names = [
        'before_dm_1', 'dm_to_in_1', 'in_to_out_1', 'after_out_1',
        'out1_to_dm2',
        'before_dm_2', 'dm_to_in_2', 'in_to_out_2', 'after_out_2'
    ]

    for seg, seg_name in zip(segments, segment_names):
        warped_seg = linear_warp(seg, windows[seg_name], sampling_rate)
        if np.any(np.isnan(warped_seg)):
            return None, None
        warped_segments.append(warped_seg)

    warped_signal = np.concatenate(warped_segments)

    cumulative_idx = 0
    segment_indices = {}

    for seg_name, warped_seg in zip(segment_names, warped_segments):
        segment_indices[f'{seg_name}_start'] = cumulative_idx
        cumulative_idx += len(warped_seg)
        segment_indices[f'{seg_name}_end'] = cumulative_idx

    segment_indices['dm_1_idx'] = segment_indices['before_dm_1_end']
    segment_indices['in_1_idx'] = segment_indices['dm_to_in_1_end']
    segment_indices['out_1_idx'] = segment_indices['in_to_out_1_end']
    segment_indices['dm_2_idx'] = segment_indices['before_dm_2_end']
    segment_indices['in_2_idx'] = segment_indices['dm_to_in_2_end']
    segment_indices['out_2_idx'] = segment_indices['in_to_out_2_end']

    return warped_signal, segment_indices


def apply_normalization(signal_matrix, config):
    """应用信号标准化（支持一维和二维输入）"""
    norm_config = config['normalization']

    if signal_matrix.ndim == 1:
        low = np.nanpercentile(signal_matrix, norm_config['lower_q'])
        high = np.nanpercentile(signal_matrix, norm_config['upper_q'])
    elif signal_matrix.ndim == 2:
        low = np.nanpercentile(signal_matrix, norm_config['lower_q'], axis=1, keepdims=True)
        high = np.nanpercentile(signal_matrix, norm_config['upper_q'], axis=1, keepdims=True)
    else:
        return signal_matrix

    normalized = (signal_matrix - low) / (high - low)
    return np.clip(normalized, 0, 1)


def load_and_warp_single_trial_data(file_paths, config):
    """加载数据并进行单试验时间规整"""
    with open(file_paths['input_data'], 'rb') as f:
        dataset = pickle.load(f)

    processed_data = []
    stats = {'total_trials': 0, 'valid_trials': 0, 'rejected_trials': 0,
             'trials_with_next_segment': 0, 'trials_with_prev_segment': 0}
    windows = config['trial_windows']

    dataset = dataset.sort_values(['mouse_id', 'phase_id', 'trial_id'])

    for idx, trial in dataset.iterrows():
        stats['total_trials'] += 1

        next_trial_data = None
        next_trials = dataset[
            (dataset['mouse_id'] == trial['mouse_id']) &
            (dataset['phase_id'] == trial['phase_id']) &
            (dataset['brain_region'] == trial['brain_region']) &
            (dataset['trial_id'] == trial['trial_id'] + 1)
            ]
        if not next_trials.empty:
            next_trial_data = next_trials.iloc[0]

        prev_trial_data = None
        prev_trials = dataset[
            (dataset['mouse_id'] == trial['mouse_id']) &
            (dataset['phase_id'] == trial['phase_id']) &
            (dataset['brain_region'] == trial['brain_region']) &
            (dataset['trial_id'] == trial['trial_id'] - 1)
            ]
        if not prev_trials.empty:
            prev_trial_data = prev_trials.iloc[0]

        warped_signal, warped_out_to_next_in, warped_prev_out_to_curr_in = create_warped_time(
            trial, next_trial_data, prev_trial_data, config
        )

        if warped_signal is None:
            stats['rejected_trials'] += 1
            continue

        warped_signal_normed = apply_normalization(warped_signal, config)

        warped_out_to_next_in_normed = None
        if warped_out_to_next_in is not None:
            warped_out_to_next_in_normed = apply_normalization(warped_out_to_next_in, config)
            stats['trials_with_next_segment'] += 1

        warped_prev_out_to_curr_in_normed = None
        if warped_prev_out_to_curr_in is not None:
            warped_prev_out_to_curr_in_normed = apply_normalization(warped_prev_out_to_curr_in, config)
            stats['trials_with_prev_segment'] += 1

        stats['valid_trials'] += 1

        if align_point == 'i_dm':
            dm_time_rel_idx = int(windows['before_dm'] * config['sampling_rate'])
            i_time_rel_idx = int((windows['before_dm'] + windows['dm_to_in']) * config['sampling_rate'])
            o_time_rel_idx = int(
                (windows['before_dm'] + windows['dm_to_in'] + windows['middle']) * config['sampling_rate'])

            trial_processed = {
                'mouse_id': trial['mouse_id'],
                'phase_id': trial['phase_id'],
                'trial_id': trial['trial_id'],
                'warped_signal': warped_signal,
                'warped_signal_normed': warped_signal_normed,
                'warped_out_to_next_in_signal': warped_out_to_next_in,
                'warped_out_to_next_in_signal_normed': warped_out_to_next_in_normed,
                'warped_prev_out_to_curr_in_signal': warped_prev_out_to_curr_in,
                'warped_prev_out_to_curr_in_signal_normed': warped_prev_out_to_curr_in_normed,
                't_start': trial['t_time_start'],
                't_end': trial['t_time_end'],
                'dm_time': trial['dm_time'],
                'i_time': trial['i_time'],
                'o_time': trial['o_time'],
                'dm_time_rel_idx': dm_time_rel_idx,
                'i_time_rel_idx': i_time_rel_idx,
                'o_time_rel_idx': o_time_rel_idx,
                'original_df_f': trial['df_f'],
                'brain_region': trial['brain_region'],
                'current_choice': trial['current_choice'],
                'next_choice': trial['next_choice'],
                'prev_choice': trial['prev_choice'],
            }
        else:
            if align_point == 'i':
                align_time_rel_idx = int(windows['before_in'] * config['sampling_rate'])
                o_time_rel_idx = int((windows['before_in'] + windows['middle']) * config['sampling_rate'])
            else:
                align_time_rel_idx = int(windows['before_in'] * config['sampling_rate'])
                o_time_rel_idx = int((windows['before_in'] + windows['middle']) * config['sampling_rate'])

            trial_processed = {
                'mouse_id': trial['mouse_id'],
                'phase_id': trial['phase_id'],
                'trial_id': trial['trial_id'],
                'warped_signal': warped_signal,
                'warped_signal_normed': warped_signal_normed,
                'warped_out_to_next_in_signal': None,
                'warped_out_to_next_in_signal_normed': None,
                'warped_prev_out_to_curr_in_signal': None,
                'warped_prev_out_to_curr_in_signal_normed': None,
                't_start': trial['t_time_start'],
                't_end': trial['t_time_end'],
                'dm_time': trial['dm_time'],
                'i_time': trial['i_time'],
                'o_time': trial['o_time'],
                f'{align_point}_time_rel_idx': align_time_rel_idx,
                'o_time_rel_idx': o_time_rel_idx,
                'original_df_f': trial['df_f'],
                'brain_region': trial['brain_region'],
                'current_choice': trial['current_choice'],
                'next_choice': trial['next_choice'],
                'prev_choice': trial['prev_choice'],
            }

        processed_data.append(trial_processed)

    print(f"数据处理完成: 总试验数={stats['total_trials']}, "
          f"有效试验数={stats['valid_trials']}, 拒绝试验数={stats['rejected_trials']}, "
          f"包含out_to_next_in段的试验数={stats['trials_with_next_segment']}, "
          f"包含prev_out_to_curr_in段的试验数={stats['trials_with_prev_segment']}")

    return pd.DataFrame(processed_data), stats


def load_and_warp_cross_trial_data(file_paths, config):
    """加载数据并进行跨试验时间规整"""
    with open(file_paths['input_data'], 'rb') as f:
        dataset = pickle.load(f)

    print(f"原始数据集大小: {len(dataset)} trials")

    dataset = dataset.sort_values(['mouse_id', 'brain_region', 'phase_id', 'trial_id'])

    processed_data = []
    stats = {
        'total_pairs': 0,
        'valid_pairs': 0,
        'rejected_pairs': 0
    }

    for (mouse_id, brain_region, phase_id), group in dataset.groupby(['mouse_id', 'brain_region', 'phase_id']):
        trials = group.sort_values('trial_id').reset_index(drop=True)

        for i in range(len(trials) - 1):
            stats['total_pairs'] += 1

            trial_1 = trials.iloc[i]
            trial_2 = trials.iloc[i + 1]

            warped_signal, segment_indices = create_cross_trial_warped_signal(
                trial_1, trial_2, config
            )

            if warped_signal is None:
                stats['rejected_pairs'] += 1
                continue

            stats['valid_pairs'] += 1

            trial_processed = {
                'mouse_id': trial_1['mouse_id'],
                'brain_region': trial_1['brain_region'],
                'phase_id': trial_1['phase_id'],
                'trial_id_1': trial_1['trial_id'],
                'trial_id_2': trial_2['trial_id'],
                'dm_time_1': trial_1['dm_time'],
                'i_time_1': trial_1['i_time'],
                'o_time_1': trial_1['o_time'],
                't_time_start_1': trial_1['t_time_start'],
                't_time_end_1': trial_1['t_time_end'],
                'dm_time_2': trial_2['dm_time'],
                'i_time_2': trial_2['i_time'],
                'o_time_2': trial_2['o_time'],
                't_time_start_2': trial_2['t_time_start'],
                't_time_end_2': trial_2['t_time_end'],
                'current_choice_1': trial_1['current_choice'],
                'next_choice_1': trial_1['next_choice'],
                'prev_choice_1': trial_1['prev_choice'],
                'current_choice_2': trial_2['current_choice'],
                'next_choice_2': trial_2['next_choice'],
                'prev_choice_2': trial_2['prev_choice'],
                'original_df_f_1': trial_1['df_f'],
                'original_df_f_2': trial_2['df_f'],
                'cross_trial_warped_signal': warped_signal,
                'segment_indices': segment_indices,
                'warped_signal_length': len(warped_signal),
                'expected_length': sum([
                    int(config['cross_trial_windows'][k] * config['sampling_rate'])
                    for k in config['cross_trial_windows'].keys()
                ])
            }

            processed_data.append(trial_processed)

    print(f"\n处理统计:")
    print(f"  总trial对数: {stats['total_pairs']}")
    print(f"  有效trial对数: {stats['valid_pairs']}")
    print(f"  拒绝trial对数: {stats['rejected_pairs']}")
    if stats['total_pairs'] > 0:
        print(f"  成功率: {stats['valid_pairs'] / stats['total_pairs'] * 100:.2f}%")

    result_df = pd.DataFrame(processed_data)

    if len(result_df) > 0:
        print(f"\n数据集统计:")
        print(f"  小鼠数量: {result_df['mouse_id'].nunique()}")
        print(f"  脑区: {result_df['brain_region'].unique()}")
        print(f"  阶段: {result_df['phase_id'].unique()}")
        print(f"  Warped信号长度: {result_df['warped_signal_length'].iloc[0]} 个时间点")
        print(f"  期望长度: {result_df['expected_length'].iloc[0]} 个时间点")

    return result_df, stats


def organize_trials_by_choice_and_phase(df, hmm_df, config):
    """按选择类型和实验阶段组织试验数据"""
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    phase_order = config['phase_order']

    for mouse_id in df['mouse_id'].unique():
        mouse_trials = df[df['mouse_id'] == mouse_id]

        for phase_name in phase_order:
            phase_trials = mouse_trials[mouse_trials['phase_id'] == phase_name]
            phase_trials = phase_trials.sort_values('trial_id')

            for _, trial_row in phase_trials.iterrows():
                hmm_subset = hmm_df[
                    (hmm_df['mouse_id'] == mouse_id) &
                    (hmm_df['phase_id'] == phase_name) &
                    (hmm_df['trial_id'] == trial_row['trial_id'])
                    ]

                if align_point == 'i_dm':
                    trial_info = {
                        'trial_id': trial_row['trial_id'],
                        'phase_id': trial_row['phase_id'],
                        'choice_type': trial_row['current_choice'],
                        'warped_signal': np.array(trial_row['warped_signal']),
                        'warped_out_to_next_in_signal': trial_row['warped_out_to_next_in_signal'],
                        'warped_prev_out_to_curr_in_signal': trial_row['warped_prev_out_to_curr_in_signal'],
                        'original_trial_data': trial_row,
                        'dm_time_rel_idx': trial_row['dm_time_rel_idx'],
                        'i_time_rel_idx': trial_row['i_time_rel_idx'],
                        'o_time_rel_idx': trial_row['o_time_rel_idx'],
                        'hmm_data': {
                            'win_probs': hmm_subset['win_probs'].iloc[0] if not hmm_subset.empty else 0,
                            'go_probs': hmm_subset['go_probs'].iloc[0] if not hmm_subset.empty else 0,
                            'random_probs': hmm_subset['random_probs'].iloc[0] if not hmm_subset.empty else 0
                        }
                    }
                else:
                    trial_info = {
                        'trial_id': trial_row['trial_id'],
                        'phase_id': trial_row['phase_id'],
                        'choice_type': trial_row['current_choice'],
                        'warped_signal': np.array(trial_row['warped_signal']),
                        'warped_out_to_next_in_signal': None,
                        'warped_prev_out_to_curr_in_signal': None,
                        'original_trial_data': trial_row,
                        f'{align_point}_time_rel_idx': trial_row[f'{align_point}_time_rel_idx'],
                        'o_time_rel_idx': trial_row['o_time_rel_idx'],
                        'hmm_data': {
                            'win_probs': hmm_subset['win_probs'].iloc[0] if not hmm_subset.empty else 0,
                            'go_probs': hmm_subset['go_probs'].iloc[0] if not hmm_subset.empty else 0,
                            'random_probs': hmm_subset['random_probs'].iloc[0] if not hmm_subset.empty else 0
                        }
                    }

                choice_type = trial_row['current_choice']
                threshold = 20 if phase_name == 'baseline' else 30
                if len(organized_data['all'][mouse_id][phase_name]) < threshold:
                    organized_data['all'][mouse_id][phase_name].append(trial_info)
                    if choice_type == 'w':
                        organized_data['w'][mouse_id][phase_name].append(trial_info)
                    elif choice_type == 'g':
                        organized_data['g'][mouse_id][phase_name].append(trial_info)

    return dict(organized_data)


def get_max_trials_per_phase_across_mice(organized_data, trial_type, config):
    """获取各个阶段的最大试验数"""
    max_trials_per_phase = {}
    phase_order = config['phase_order']

    for phase_name in phase_order:
        max_trials = 0
        for mouse_id, mouse_data in organized_data[trial_type].items():
            if phase_name in mouse_data:
                phase_trial_count = len(mouse_data[phase_name])
                max_trials = max(max_trials, phase_trial_count)
        max_trials_per_phase[phase_name] = max_trials

    return max_trials_per_phase


def interpolate_trial_positions(num_trials, max_trials):
    """插值试验位置"""
    if max_trials <= 1:
        return [0] * num_trials
    positions = np.linspace(0, max_trials - 1, num_trials)
    return [int(round(pos)) for pos in positions]


def calculate_choice_proportions_by_phase(mouse_data, trial_type, config):
    """计算各阶段的选择比例"""
    phase_order = config['phase_order']
    phase_proportions = []
    phase_counts = []

    for phase_name in phase_order:
        if phase_name in mouse_data:
            trials = mouse_data[phase_name]
            if trial_type == 'all':
                win_count = sum(1 for t in trials if t['choice_type'] == 'w')
                total_count = len(trials)
                proportion = win_count / total_count if total_count > 0 else 0.0
            elif trial_type == 'w':
                proportion = 1.0 if len(trials) > 0 else 0.0
                total_count = len(trials)
            elif trial_type == 'g':
                proportion = 0.0 if len(trials) > 0 else 0.0
                total_count = len(trials)

            phase_proportions.append(proportion)
            phase_counts.append(total_count)
        else:
            phase_proportions.append(0.0)
            phase_counts.append(0)

    return np.array(phase_proportions), np.array(phase_counts)


def create_layout_params(organized_data, trial_type, config, is_population=True):
    """创建布局参数"""
    phase_order = config['phase_order']

    if is_population:
        max_trials_per_phase = get_max_trials_per_phase_across_mice(organized_data, trial_type, config)
    else:
        max_trials_per_phase = {}
        for phase_name in phase_order:
            first_key = next(iter(organized_data[trial_type]))
            max_trials_per_phase[phase_name] = len(
                organized_data[trial_type][first_key].get(phase_name, [])
            )

    phase_boundaries = [0]
    phase_centers = []
    cumulative_trials = 0

    for phase_name in phase_order:
        phase_trial_count = max_trials_per_phase[phase_name]
        cumulative_trials += phase_trial_count
        phase_boundaries.append(cumulative_trials)
        phase_center = phase_boundaries[-2] + phase_trial_count / 2
        phase_centers.append(phase_center)

    return {
        'phase_boundaries': phase_boundaries,
        'phase_centers': phase_centers,
        'total_rows': cumulative_trials,
        'max_trials_per_phase': max_trials_per_phase
    }


def aggregate_hmm_data(hmm_posteriors_collection):
    """聚合HMM数据"""
    records = []
    for row_idx, dict_list in hmm_posteriors_collection.items():
        for d in dict_list:
            rec = d.copy()
            rec["row_idx"] = row_idx
            records.append(rec)

    if not records:
        return {}

    df = pd.DataFrame(records)
    agg_df = df.groupby("row_idx").agg(["mean", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0])
    agg_df = agg_df.rename(columns={"<lambda_0>": "sem"})

    hmm_result = {
        idx: {
            col: {"mean": row[(col, "mean")], "sem": row[(col, "sem")]}
            for col in ["win_probs", "go_probs", "random_probs"]
        }
        for idx, row in agg_df.iterrows()
    }

    return hmm_result


def create_single_mouse_matrices(mouse_id, mouse_data, trial_type, config):
    """创建单只小鼠的矩阵"""
    phase_order = config['phase_order']

    all_trial_signals = []
    all_out_to_next_in_signals = []
    all_prev_out_to_curr_in_signals = []
    trial_ids = []
    hmm_posteriors = []
    choices = []

    for phase_name in phase_order:
        if phase_name in mouse_data:
            phase_trials = mouse_data[phase_name]
            for trial_data in phase_trials:
                all_trial_signals.append(trial_data['warped_signal'])
                all_out_to_next_in_signals.append(trial_data['warped_out_to_next_in_signal'])
                all_prev_out_to_curr_in_signals.append(trial_data['warped_prev_out_to_curr_in_signal'])
                trial_ids.append(trial_data['trial_id'])
                hmm_posteriors.append(trial_data['hmm_data'])
                choices.append(trial_data['choice_type'])


    signal_matrix = np.array(all_trial_signals)
    normalized_signal_matrix = apply_normalization(signal_matrix, config)

    trial_proportions, trial_counts = calculate_choice_proportions_by_phase(mouse_data, trial_type, config)

    layout_params = create_layout_params({trial_type: {mouse_id: mouse_data}}, trial_type, config, is_population=False)

    first_trial = None
    for phase_name in phase_order:
        if phase_name in mouse_data and mouse_data[phase_name]:
            first_trial = mouse_data[phase_name][0]
            break

    if first_trial:
        if align_point == 'i_dm':
            layout_params['dm_time_rel_idx'] = first_trial['dm_time_rel_idx']
            layout_params['i_time_rel_idx'] = first_trial['i_time_rel_idx']
            layout_params['o_time_rel_idx'] = first_trial['o_time_rel_idx']
        else:
            layout_params[f'{align_point}_time_rel_idx'] = first_trial[f'{align_point}_time_rel_idx']
            layout_params['o_time_rel_idx'] = first_trial['o_time_rel_idx']

    return {
        'mouse_id': mouse_id,
        'trial_type': trial_type,
        'avg_matrix': normalized_signal_matrix,
        'avg_matrix_raw': signal_matrix,
        'out_to_next_in_signals': all_out_to_next_in_signals,
        'prev_out_to_curr_in_signals': all_prev_out_to_curr_in_signals,
        'trial_ids': trial_ids,
        'proportions': trial_proportions,
        'counts': trial_counts,
        'hmm_posteriors': hmm_posteriors,
        'layout_params': layout_params,
        'choices': choices
    }


def create_population_average_matrices(organized_data, trial_type, config):
    """创建群体平均矩阵"""
    phase_order = config['phase_order']
    layout_params = create_layout_params(organized_data, trial_type, config, is_population=True)

    position_signal_collection = defaultdict(list)
    raw_signal_collection = defaultdict(list)
    out_to_next_in_collection = defaultdict(list)
    prev_out_to_curr_in_collection = defaultdict(list)
    hmm_posteriors_collection = defaultdict(list)
    win_choice_collection = {}

    for mouse_id, mouse_data in organized_data[trial_type].items():
        global_row_offset = 0

        for phase_idx, phase_name in enumerate(phase_order):
            max_phase_trials = layout_params['max_trials_per_phase'][phase_name]

            if phase_name in mouse_data:
                phase_trials = mouse_data[phase_name]
                num_trials = len(phase_trials)
                if trial_type != 'all':
                    trial_positions = interpolate_trial_positions(num_trials, max_phase_trials)
                else:
                    trial_positions = list(range(num_trials))

                phase_signals = [trial['warped_signal'] for trial in phase_trials]
                hmm_posteriors = [trial['hmm_data'] for trial in phase_trials]
                out_to_next_in_signals = [trial['warped_out_to_next_in_signal'] for trial in phase_trials]
                prev_out_to_curr_in_signals = [trial['warped_prev_out_to_curr_in_signal'] for trial in phase_trials]

                if phase_signals:
                    phase_signal_matrix = np.array(phase_signals)
                    normalized_phase_signals = apply_normalization(phase_signal_matrix, config)

                    for trial_idx, (normalized_signal, raw_signal, out_to_next_in_sig,
                                    prev_out_to_curr_in_sig) in enumerate(
                            zip(normalized_phase_signals, phase_signal_matrix, out_to_next_in_signals,
                                prev_out_to_curr_in_signals)):
                        if trial_idx < len(trial_positions):
                            aligned_row = global_row_offset + trial_positions[trial_idx]

                            position_signal_collection[aligned_row].append(normalized_signal)
                            raw_signal_collection[aligned_row].append(raw_signal)
                            hmm_posteriors_collection[aligned_row].append(hmm_posteriors[trial_idx])

                            if out_to_next_in_sig is not None:
                                out_to_next_in_collection[aligned_row].append(out_to_next_in_sig)

                            if prev_out_to_curr_in_sig is not None:
                                prev_out_to_curr_in_collection[aligned_row].append(prev_out_to_curr_in_sig)

                            if aligned_row not in win_choice_collection:
                                win_choice_collection[aligned_row] = 0
                            if phase_trials[trial_idx]['choice_type'] == 'w':
                                win_choice_collection[aligned_row] += 1

            global_row_offset += max_phase_trials

    hmm_result = aggregate_hmm_data(hmm_posteriors_collection)

    total_rows = layout_params['total_rows']
    if not position_signal_collection:
        return None

    signal_length = len(list(position_signal_collection.values())[0][0])
    average_signal_matrix = np.full((total_rows, signal_length), np.nan)
    raw_signal_matrix = np.full((total_rows, signal_length), np.nan)
    trial_count_by_position = np.zeros(total_rows)

    out_to_next_in_length = config['trial_windows']['out_to_next_in'] * config['sampling_rate']
    out_to_next_in_matrix = np.full((total_rows, int(out_to_next_in_length)), np.nan)

    prev_out_to_curr_in_length = config['trial_windows']['prev_out_to_curr_in'] * config['sampling_rate']
    prev_out_to_curr_in_matrix = np.full((total_rows, int(prev_out_to_curr_in_length)), np.nan)

    for row_idx in range(total_rows):
        if row_idx in position_signal_collection and position_signal_collection[row_idx]:
            position_signals = np.array(position_signal_collection[row_idx])
            raw_signals = np.array(raw_signal_collection[row_idx])

            average_signal_matrix[row_idx, :] = np.mean(position_signals, axis=0)
            raw_signal_matrix[row_idx, :] = np.mean(raw_signals, axis=0)

            trial_count_by_position[row_idx] = len(position_signal_collection[row_idx])

        if row_idx in out_to_next_in_collection and out_to_next_in_collection[row_idx]:
            valid_out_to_next_in = [sig for sig in out_to_next_in_collection[row_idx] if sig is not None]
            if valid_out_to_next_in:
                out_to_next_in_signals_array = np.array(valid_out_to_next_in)
                out_to_next_in_matrix[row_idx, :] = np.mean(out_to_next_in_signals_array, axis=0)

        if row_idx in prev_out_to_curr_in_collection and prev_out_to_curr_in_collection[row_idx]:
            valid_prev_out_to_curr_in = [sig for sig in prev_out_to_curr_in_collection[row_idx] if sig is not None]
            if valid_prev_out_to_curr_in:
                prev_out_to_curr_in_signals_array = np.array(valid_prev_out_to_curr_in)
                prev_out_to_curr_in_matrix[row_idx, :] = np.mean(prev_out_to_curr_in_signals_array, axis=0)

    average_signal_matrix = apply_normalization(average_signal_matrix, config)

    # 在average_signal_matrix, raw_signal_matrix计算之后，增加SEM计算
    sem_signal_matrix = np.full((total_rows, signal_length), np.nan)
    raw_sem_matrix = np.full((total_rows, signal_length), np.nan)
    out_to_next_in_sem_matrix = np.full((total_rows, int(out_to_next_in_length)), np.nan)
    prev_out_to_curr_in_sem_matrix = np.full((total_rows, int(prev_out_to_curr_in_length)), np.nan)

    for row_idx in range(total_rows):
        if row_idx in position_signal_collection and position_signal_collection[row_idx]:
            position_signals = np.array(position_signal_collection[row_idx])
            sem_signal_matrix[row_idx, :] = np.std(position_signals, axis=0, ddof=1) / np.sqrt(len(position_signals))

            raw_signals = np.array(raw_signal_collection[row_idx])
            raw_sem_matrix[row_idx, :] = np.std(raw_signals, axis=0, ddof=1) / np.sqrt(len(raw_signals))

        if row_idx in out_to_next_in_collection and out_to_next_in_collection[row_idx]:
            valid_out_to_next_in = [sig for sig in out_to_next_in_collection[row_idx] if sig is not None]
            if valid_out_to_next_in:
                out_to_next_in_signals_array = np.array(valid_out_to_next_in)
                out_to_next_in_sem_matrix[row_idx, :] = np.std(out_to_next_in_signals_array, axis=0, ddof=1) / np.sqrt(
                    len(valid_out_to_next_in))

        if row_idx in prev_out_to_curr_in_collection and prev_out_to_curr_in_collection[row_idx]:
            valid_prev_out_to_curr_in = [sig for sig in prev_out_to_curr_in_collection[row_idx] if sig is not None]
            if valid_prev_out_to_curr_in:
                prev_out_to_curr_in_signals_array = np.array(valid_prev_out_to_curr_in)
                prev_out_to_curr_in_sem_matrix[row_idx, :] = np.std(prev_out_to_curr_in_signals_array, axis=0,
                                                                    ddof=1) / np.sqrt(len(valid_prev_out_to_curr_in))

    all_mouse_data = organized_data[trial_type]
    avg_proportions_by_phase = []
    avg_counts_by_phase = []

    for phase_name in phase_order:
        phase_proportions = []
        phase_counts = []

        for mouse_id, mouse_data in all_mouse_data.items():
            proportions, counts = calculate_choice_proportions_by_phase(mouse_data, trial_type, config)
            phase_idx = phase_order.index(phase_name)
            if phase_idx < len(proportions) and counts[phase_idx] > 0:
                phase_proportions.append(proportions[phase_idx])
                phase_counts.append(counts[phase_idx])

        if phase_proportions:
            avg_proportions_by_phase.append(np.mean(phase_proportions))
            avg_counts_by_phase.append(np.mean(phase_counts))
        else:
            avg_proportions_by_phase.append(0)
            avg_counts_by_phase.append(0)

    win_choice_by_position = np.array([win_choice_collection.get(i, 0) for i in range(total_rows)])
    choose_proportion_by_position = win_choice_by_position / (trial_count_by_position + 1e-6)

    first_trial = None
    for mouse_id, mouse_data in organized_data[trial_type].items():
        for phase_name in config['phase_order']:
            if phase_name in mouse_data and mouse_data[phase_name]:
                first_trial = mouse_data[phase_name][0]
                break
        if first_trial:
            break

    if first_trial:
        if align_point == 'i_dm':
            layout_params['dm_time_rel_idx'] = first_trial['dm_time_rel_idx']
            layout_params['i_time_rel_idx'] = first_trial['i_time_rel_idx']
            layout_params['o_time_rel_idx'] = first_trial['o_time_rel_idx']
        else:
            layout_params[f'{align_point}_time_rel_idx'] = first_trial[f'{align_point}_time_rel_idx']
            layout_params['o_time_rel_idx'] = first_trial['o_time_rel_idx']

    return {
        'trial_type': trial_type,
        'avg_matrix': average_signal_matrix,
        'raw_matrix': raw_signal_matrix,
        'sem_matrix': sem_signal_matrix,
        'raw_sem_matrix': raw_sem_matrix,
        'out_to_next_in_matrix': out_to_next_in_matrix,
        'out_to_next_in_sem_matrix': out_to_next_in_sem_matrix,
        'prev_out_to_curr_in_matrix': prev_out_to_curr_in_matrix,
        'prev_out_to_curr_in_sem_matrix': prev_out_to_curr_in_sem_matrix,
        'proportions': np.array(avg_proportions_by_phase),
        'counts': np.array(avg_counts_by_phase),
        'hmm_posteriors': hmm_result,
        'layout_params': layout_params,
        'trial_count_by_position': trial_count_by_position,
        'win_choice_by_position': win_choice_by_position,
        'choose_proportion_by_position': choose_proportion_by_position
    }


def generate_matrices_for_brain_region(organized_data, brain_region, file_paths, config):
    """为特定脑区生成所有矩阵"""
    trial_types = ['all','w','g']

    results = {
        'individual_mice': {},
        'population_averages': {},
    }

    for trial_type in trial_types:
        print(f"处理 {brain_region} 脑区的 {trial_type} 试验...")

        results['individual_mice'][trial_type] = {}
        for mouse_id, mouse_data in organized_data[trial_type].items():
            mouse_result = create_single_mouse_matrices(mouse_id, mouse_data, trial_type, config)
            if mouse_result is not None:
                results['individual_mice'][trial_type][mouse_id] = mouse_result

        pop_result = create_population_average_matrices(organized_data, trial_type, config)
        if pop_result is not None:
            results['population_averages'][trial_type] = pop_result

    output_path = file_paths['matrix_output_template'].format(brain_region)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"{brain_region} 脑区矩阵已保存到 {output_path}")
    return results


def main():
    file_paths = get_file_paths(CONFIG, align_point)

    if align_point == 'cross_trial':
        print("\n=== 跨试验时间规整模式 ===")
        print("\nWarping配置:")
        total_points = 0
        for key, value in CONFIG['cross_trial_windows'].items():
            points = int(value * CONFIG['sampling_rate'])
            total_points += points
            print(f"  {key}: {value}秒 ({points}个点)")
        print(f"  总长度: {total_points}个时间点 ({total_points / CONFIG['sampling_rate']}秒)")
        print("=" * 60)

        warped_data, processing_stats = load_and_warp_cross_trial_data(file_paths, CONFIG)

        os.makedirs(os.path.dirname(file_paths['warped_output']), exist_ok=True)
        with open(file_paths['warped_output'], 'wb') as f:
            pickle.dump(warped_data, f)

        print(f"\n结果已保存到: {file_paths['warped_output']}")
        print("\n处理完成!")

    else:
        print(f"\n=== 单试验时间规整模式: {align_point} ===")

        warped_data, processing_stats = load_and_warp_single_trial_data(file_paths, CONFIG)

        os.makedirs(os.path.dirname(file_paths['warped_output']), exist_ok=True)
        with open(file_paths['warped_output'], 'wb') as f:
            pickle.dump(warped_data, f)

        hmm_df = pd.read_csv(file_paths['hmm_results'])

        for brain_region in CONFIG['brain_regions']:
            print(f"\n处理 {brain_region} 脑区...")

            region_data = warped_data[warped_data['brain_region'] == brain_region]

            organized_data = organize_trials_by_choice_and_phase(region_data, hmm_df, CONFIG)

            generate_matrices_for_brain_region(organized_data, brain_region, file_paths, CONFIG)

        print("\n处理完成!")


if __name__ == "__main__":
    main()