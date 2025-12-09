import numpy as np
import pandas as pd
import os
import re

### desample function ###
def merge_intervals(intervals):
    """合并重叠的区间列表 [(s,e), ...] → [(s1,e1), ...]"""
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [list(ints[0])]
    for s,e in ints[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s,e])
    return [(s,e) for s,e in merged]


def subtract_intervals(keep, remove):
    """
    从 keep_intervals 中删去所有 remove_intervals：
    keep=[(a,b),...] 删去 remove=[(u,v),...] → 新的 keep
    """
    out = []
    for ks,ke in keep:
        segments = [(ks,ke)]
        for rs,re in remove:
            new_segments = []
            for cs,ce in segments:
                # 没有交集
                if re <= cs or rs >= ce:
                    new_segments.append((cs,ce))
                else:
                    # 左侧残段
                    if rs > cs:
                        new_segments.append((cs, rs))
                    # 右侧残段
                    if re < ce:
                        new_segments.append((re, ce))
            segments = new_segments
        out.extend(segments)
    return merge_intervals(out)


def complement_intervals(intervals, t0, t1):
    """
    取 [t0,t1] 区间减去 intervals → 返回补集区间列表
    """
    if not intervals:
        return [(t0, t1)]
    ints = merge_intervals(intervals)
    out = []
    # 前缀
    if ints[0][0] > t0:
        out.append((t0, ints[0][0]))
    # 中间缝隙
    for (a,b),(c,d) in zip(ints, ints[1:]):
        if b < c:
            out.append((b,c))
    # 后缀
    if ints[-1][1] < t1:
        out.append((ints[-1][1], t1))
    return out

# ---------- 1) 从 z6 提取连续段 ----------
def extract_segments_with_indices(z6, label_map):
    """
    从一维标签序列 z6 提取连续段，返回 DataFrame，包含：
    start_idx, end_idx（半开 [s,e)）, start_time, end_time, label, label_name
    在“索引即时间”的设定下，start_time=end_idx 与 start_idx/end_idx 一致。
    """
    z6 = np.asarray(z6, dtype=int)
    if z6.size == 0:
        return pd.DataFrame(columns=["start_idx","end_idx","start_time","end_time","label","label_name"])

    segs = []
    s = 0
    cur = z6[0]
    for i in range(1, z6.size):
        if z6[i] != cur:
            e = i
            segs.append((s, e, int(cur)))
            s = i
            cur = z6[i]
    segs.append((s, z6.size, int(cur)))  # 最后一段

    df = pd.DataFrame(segs, columns=["start_idx","end_idx","label"])
    # 在索引当时间的设定下，直接复用
    df["start_time"] = df["start_idx"]
    df["end_time"]   = df["end_idx"]
    df["label_name"] = df["label"].map(lambda x: label_map.get(x, str(x)))
    return df


def save_behavior_and_latents(seg_df, xhat, save_dir,
                              fname_segments="behavior_segments.csv",
                              fname_latents="latent_variables.csv"):
    """
    保存行为分段表 seg_df 和连续隐变量 xhat 为 csv
    seg_df: DataFrame, 必须包含 label_name, start_idx, end_idx 等
    xhat: (T, d) numpy 数组，连续隐变量
    save_dir: 输出目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存行为分段表
    seg_path = os.path.join(save_dir, fname_segments)
    seg_df.to_csv(seg_path, index=False)
    print(f"[Saved] behavior segments -> {seg_path}")

    # 保存隐变量
    T, d = xhat.shape
    df_latent = pd.DataFrame(xhat, columns=[f"latent_{i+1}" for i in range(d)])
    df_latent.insert(0, "time_idx", np.arange(T))  # 可选：加时间索引
    lat_path = os.path.join(save_dir, fname_latents)
    df_latent.to_csv(lat_path, index=False)
    print(f"[Saved] latent variables -> {lat_path}")
    
def save_behavior_lds_results(results_per_behavior, save_dir, filename="behavior_lds_params.csv"):
    os.makedirs(save_dir, exist_ok=True)

    records = []
    for beh, params in results_per_behavior.items():
        A = params["A"].flatten()
        c = params["c"].flatten()
        eigA = np.array(params["eigA"]).flatten()

        record = {
            "behavior": beh,
            # 展平矩阵
            "A11": A[0], "A12": A[1], "A21": A[2], "A22": A[3],
            "c1": c[0], "c2": c[1],
            "eig1": eigA[0], "eig2": eigA[1]
        }
        records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(save_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df

def _taus_from_eigs(eigs):
    """根据 A 的特征值计算离散时间常数 tau = -1 / ln|lambda|（dt=1 的等效；若有真实 dt 请再乘 dt）"""
    if eigs is None:
        return [np.nan, np.nan]
    lam_abs = np.clip(np.abs(eigs), 1e-12, None)
    taus = -1.0 / np.log(lam_abs)
    taus = np.sort(np.real(taus))
    return [taus[0], taus[-1]]  # [fast, slow]

def export_win_go_dynamics(results, save_dir,
                           group_csv="win_go_dynamics_group.csv",
                           pertrial_csv="win_go_kinematics_per_trial.csv"):
    """
    将 analyze_trials_ring_dynamics 的 outputs 导出：
      1) group 级参数 (A, c, eigvals, taus, mean speed/omega, n_trials)
      2) per-trial 运动学 (speed, angular velocity)
      3) 完整矩阵结构另存 json 便于后续载入
    """
    os.makedirs(save_dir, exist_ok=True)

    # -------- group 级：A, c, eigvals, taus, mean speed/omega --------
    rows = []

    for kind in ["go", "win"]:
        A = results[f"A_{kind}"]
        c = results[f"c_{kind}"]
        lam = results[f"eig_{kind}"]
        tau_fast, tau_slow = _taus_from_eigs(lam)

        # trial级分布（用于 group 表中的均值/SEM 与 n）
        v_all = results[f"v_{kind}_all"]
        w_all = results[f"w_{kind}_all"]
        n_trials = int(len(v_all)) if v_all is not None else 0

        def _mean_sem(arr):
            if arr is None or len(arr) == 0:
                return np.nan, np.nan
            return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

        v_mean, v_sem = _mean_sem(v_all)
        w_mean, w_sem = _mean_sem(w_all)

        # 处理可能的 None
        if A is None: 
            A = np.full((2,2), np.nan)
        if c is None: 
            c = np.full(2, np.nan)
        if lam is None:
            lam = np.array([np.nan, np.nan], dtype=complex)

        row = {
            "kind": kind,
            "n_trials": n_trials,

            # A 展平
            "A11": float(np.real(A[0,0])), "A12": float(np.real(A[0,1])),
            "A21": float(np.real(A[1,0])), "A22": float(np.real(A[1,1])),

            # c
            "c1": float(np.real(c[0])), "c2": float(np.real(c[1])),

            # eigenvalues（实/虚/模）
            "eig1_real": float(np.real(lam[0])), "eig1_imag": float(np.imag(lam[0])), "eig1_abs": float(np.abs(lam[0])),
            "eig2_real": float(np.real(lam[1])), "eig2_imag": float(np.imag(lam[1])), "eig2_abs": float(np.abs(lam[1])),

            # taus
            "tau_fast": float(tau_fast), "tau_slow": float(tau_slow),

            # 运动学（trial 分布 → group 均值±SEM）
            "speed_mean": v_mean, "speed_sem": v_sem,
            "omega_mean": w_mean, "omega_sem": w_sem,
        }
        rows.append(row)

    df_group = pd.DataFrame(rows)
    path_group = os.path.join(save_dir, group_csv)
    df_group.to_csv(path_group, index=False)
    print(f"[Saved] {path_group}")

    # -------- per-trial 级：每个 trial 的速度与角速度 --------
    recs = []
    for kind in ["go", "win"]:
        v_all = results[f"v_{kind}_all"]
        w_all = results[f"w_{kind}_all"]
        n = len(v_all) if v_all is not None else 0
        for i in range(n):
            recs.append({
                "kind": kind,
                "trial_idx": i,
                "mean_speed": float(v_all[i]),
                "mean_omega": float(w_all[i])
            })
    df_trials = pd.DataFrame(recs)
    path_trials = os.path.join(save_dir, pertrial_csv)
    df_trials.to_csv(path_trials, index=False)
    print(f"[Saved] {path_trials}")

    return dict(group_csv=path_group, pertrial_csv=path_trials)

def save_dtw_matrices_to_csv(
    D_sorted, labels_sorted, label_order, 
    name_map,              # 例如 {0:"win_center", ...}
    save_dir,
    dtw_matrix_csv="DTW_matrix_labeled.csv",
    dtw_group_csv="DTW_groupwise_mean.csv"
):
    """
    保存两个 CSV：
    1) 带行列标签（行为+序号）的 DTW 全矩阵
    2) 行列为行为类型的“分组平均距离”方阵
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 1) 带标签的全矩阵 ----------
    # 先把每个样本的行为名准备好
    labels_txt = np.array([name_map[int(l)] for l in labels_sorted], dtype=object)

    # 为每个行为名加“#序号”避免重名（如 win_tube#01, win_tube#02 ...）
    counters = {k:0 for k in set(labels_txt)}
    rowcol_tags = []
    for lab in labels_txt:
        counters[lab] += 1
        rowcol_tags.append(f"{lab}#{counters[lab]:02d}")
    rowcol_tags = np.array(rowcol_tags, dtype=object)

    # 构建 DataFrame 并保存
    df_dtw = pd.DataFrame(D_sorted, index=rowcol_tags, columns=rowcol_tags)
    path_matrix = os.path.join(save_dir, dtw_matrix_csv)
    df_dtw.to_csv(path_matrix, index=True)
    print(f"[Saved] DTW full matrix -> {path_matrix}")

    # ---------- 2) 行列为行为类型的分组平均距离方阵 ----------
    # 填表：对 (a,b) 不存在的组合填 NaN
    uniq_labels = list(label_order)  # 保持你给定的顺序
    beh_names = [name_map[int(k)] for k in uniq_labels]
    G = pd.DataFrame(np.nan, index=beh_names, columns=beh_names, dtype=float)

    # 这里复用你已有的 groupwise_mean_distances 结果；如果外面没有，就现算一遍：
    # from rSLDS_DTWhelper import groupwise_mean_distances
    # mean_d = groupwise_mean_distances(D_sorted, labels_sorted, label_order)

    # 注意：此函数假设你外部已经有 mean_d（字典：key=(a,b), val=均值）
    # 如果你此处没有 mean_d，就把上面的两行解开计算再用。
    try:
        mean_d  # noqa
    except NameError:
        from rSLDS_DTWhelper import groupwise_mean_distances
        mean_d = groupwise_mean_distances(D_sorted, labels_sorted, label_order)

    for (a, b), v in mean_d.items():
        if np.isfinite(v):
            G.loc[name_map[int(a)], name_map[int(b)]] = float(v)

    path_group = os.path.join(save_dir, dtw_group_csv)
    G.to_csv(path_group, index=True)
    print(f"[Saved] DTW groupwise mean matrix -> {path_group}")

    return {"dtw_full_csv": path_matrix, "dtw_group_csv": path_group}

# 自动从文件名推断 units_names 的函数
def infer_units_names(channel_files):
    units = []
    pat = re.compile(r"(?P<date>\d{6}).*?(?P<ch>Channel\d+).*?(?P<unit>Unit\d+)\.mat$", re.IGNORECASE)
    for f in channel_files:
        base = os.path.basename(f)
        m = pat.search(base)
        if not m:
            # 兜底：按空格切分再寻找关键片段
            stem = os.path.splitext(base)[0]
            parts = stem.split()
            date = next((p for p in parts if re.fullmatch(r"\d{6}", p)), None)
            ch   = next((p for p in parts if p.startswith("Channel")), None)
            unit = next((p for p in parts if p.startswith("Unit")), None)
            if not (date and ch and unit):
                raise ValueError(f"无法从文件名解析 unit 名称: {base}")
            units.append(f"SplitUnit_{date}_{ch}_{unit}")
        else:
            units.append(f"SplitUnit_{m['date']}_{m['ch']}_{m['unit']}")
    return units

# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

def _load_mat_variables(mat_path: str) -> dict:
    """
    加载 MATLAB .mat 文件中的变量为 {name: np.ndarray} 字典。
    - 优先用 scipy.io.loadmat（适配 v7 及以下）
    - 若失败或检测为 v7.3，尝试用 h5py（可选依赖）
    """
    # 先尝试 scipy
    try:
        from scipy.io import loadmat
        mdict = loadmat(mat_path)
        # 去掉 __header__ / __version__ / __globals__ 等内部键
        out = {}
        for k, v in mdict.items():
            if k.startswith("__"):
                continue
            # MATLAB 标量/列向量/行向量，尽量展平成 1D
            arr = np.asarray(v).squeeze()
            out[k] = arr
        if out:
            return out
    except Exception:
        pass

    # 再尝试 h5py（v7.3 HDF5）
    try:
        import h5py
        out = {}
        with h5py.File(mat_path, "r") as f:
            # v7.3 通常每个变量是一个 dataset
            for k in f.keys():
                dset = f[k]
                # 取值并展平为 1D
                val = np.array(dset[()]).squeeze()
                # h5py 读出的转置问题：按需要转置为 1D
                if val.ndim > 1:
                    # 若是列/行向量，尽量展平成一维
                    val = val.reshape(-1)
                out[k] = val
        if out:
            return out
    except Exception as e:
        raise RuntimeError(
            f"无法读取 MAT 文件：{mat_path}\n"
            f"已尝试 scipy.io.loadmat 和 h5py。若是 v7.3，请确保安装了 h5py。原始错误：{e}"
        )

    raise RuntimeError(f"未在 {mat_path} 中读到任何可用变量。")

def build_poisson_from_mat(
    mat_path: str,
    time_start: float,
    time_end: float,
    bin_size: float,
    include_vars: Optional[List[str]] = None,
    exclude_vars: Optional[List[str]] = None,
    sort_keys: bool = True,
    name_mapper=None,  # 可选：函数，把变量名映射为展示名
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    参数
    ----
    mat_path: data.mat 路径（单文件、多个变量）
    time_start, time_end: 分析窗口（秒）
    bin_size: 统计 bin 的宽度（秒）
    include_vars: 只保留这些变量名（白名单）；None 表示不限制
    exclude_vars: 排除这些变量名（黑名单）
    sort_keys: 是否对变量名排序，保证通道顺序稳定
    name_mapper: 可选函数 f(var_name)->display_name

    返回
    ----
    Poisson_activity: (n_channels, n_bins) 的整型计数矩阵
    time_axis: (n_bins,) 的时间轴（每个 bin 的左边界）
    unit_names: 通道/单元名列表（与 Poisson_activity 的第 0 维对应）
    """
    if time_end <= time_start:
        raise ValueError("time_end 必须大于 time_start。")
    if bin_size <= 0:
        raise ValueError("bin_size 必须为正。")

    # 读入所有变量
    var_dict = _load_mat_variables(mat_path)
    all_keys = [k for k in var_dict.keys()]

    # 白/黑名单筛选
    keys = all_keys
    if include_vars is not None:
        missing = [k for k in include_vars if k not in var_dict]
        if missing:
            raise KeyError(
                f"include_vars 中的变量不存在：{missing}\n"
                f"可用变量：{sorted(all_keys)}"
            )
        keys = include_vars[:]
    if exclude_vars is not None:
        keys = [k for k in keys if k not in set(exclude_vars)]

    # 过滤掉非一维向量
    filtered = []
    for k in keys:
        arr = np.asarray(var_dict[k]).squeeze()
        if arr.ndim == 1:
            filtered.append(k)
        else:
            print(f"[警告] 变量 {k} 不是一维向量，已跳过。形状={arr.shape}")

    if not filtered:
        raise RuntimeError(
            "筛选后没有一维向量变量可用。"
            f" 可用变量：{sorted(all_keys)}"
        )

    if sort_keys:
        filtered = sorted(filtered)

    # 生成时间轴与 bin 边界
    time_axis = np.arange(time_start, time_end, bin_size, dtype=float)
    # 右闭边界处理：最后一个边界 = 最后一个左边界 + bin_size
    bin_edges = np.concatenate([time_axis, [time_axis[-1] + bin_size]])
    n_bins = len(time_axis)
    n_channels = len(filtered)

    # 统计计数
    Poisson_activity = np.zeros((n_channels, n_bins), dtype=int)
    unit_names = []

    for i, k in enumerate(filtered):
        raw = np.asarray(var_dict[k]).squeeze().astype(float)
        # 直接使用真实发放时间（秒），无需延迟/采样率换算
        # 裁剪到分析窗口
        mask = (raw >= time_start) & (raw < time_end)
        spikes = raw[mask]
        counts, _ = np.histogram(spikes, bins=bin_edges)
        Poisson_activity[i, :] = counts

        unit_names.append(name_mapper(k) if callable(name_mapper) else k)
        print(f"[{i+1}/{n_channels}] 变量 {k}: 有效脉冲 {spikes.size} 个 -> 已计数")

    print("Activity counts shape:", Poisson_activity.shape)
    return Poisson_activity, time_axis, unit_names
