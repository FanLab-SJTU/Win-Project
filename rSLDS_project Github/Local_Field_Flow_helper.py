#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:06:47 2025

@author: shuimuqinghua
"""
import numpy as np
import pandas as pd
import os
from numpy.linalg import eigvals, lstsq
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d
import seaborn as sns


def compute_average_trajectory(xhat, segs, n_points=60):
    """
    xhat: (T,2) latent states
    segs: [(i0,i1), ...] 段索引
    n_points: 对齐到多少点
    返回：mean_traj (n_points,2), std_traj (n_points,2), progress(0~1)
    """
    resampled = []
    for (i0,i1) in segs:
        if i1 - i0 < 3: 
            continue
        seg = xhat[i0:i1]
        t = np.linspace(0,1,len(seg))
        f0 = interp1d(t, seg[:,0], kind="linear")
        f1 = interp1d(t, seg[:,1], kind="linear")
        new_t = np.linspace(0,1,n_points)
        new_seg = np.stack([f0(new_t), f1(new_t)], axis=1)
        resampled.append(new_seg)
    if not resampled:
        return None, None, None
    arr = np.stack(resampled, axis=0)   # (n_seg, n_points, 2)
    mean_traj = arr.mean(axis=0)
    std_traj = arr.std(axis=0)
    return mean_traj, std_traj, np.linspace(0,1,n_points)

def Local_Field_Flow_Features(Xseg, tseg):
    """
    Xseg: (n,2) latent states within a segment
    tseg: (n,) time stamps
    returns dict of per-segment metrics
    """
    n = len(tseg)
    # if n <= 10:  # 太短不稳
    #     return None

    dt = 0.1 #np.median(np.diff(tseg))       # step 时间中位数作为代表步长 默认=1
    V = np.diff(Xseg, axis=0) / dt      # 间隔差分 / 时间 = 速度
    v = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)  # 单位速度方向
    path_len = np.sum(np.linalg.norm(np.diff(Xseg, axis=0), axis=1))    # 几何曲线长度 路径长度
    chord = np.linalg.norm(Xseg[-1] - Xseg[0]) + 1e-9   # 首位距离
    tortuosity = path_len / chord   # Tortuosity 接近1越直线 接近0越曲线毛线团

    ang = []
    for i in range(len(v)-1):   # 平均转角（相邻速度间夹角）
        c = np.clip(np.dot(v[i], v[i+1]), -1.0, 1.0)
        ang.append(np.arccos(c))
    mean_turn = float(np.mean(ang)) if len(ang)>0 else np.nan

    # 速度 PCA 一维性：第一特征值占比
    Sv = np.cov(V.T)  # 2x2
    ev_v = np.linalg.eigvalsh(Sv)  # 升序
    ev_v = np.maximum(ev_v, 1e-12)
    evr1 = float(ev_v[-1] / ev_v.sum())  # 越接近 1 越线性

    # 轨迹参与比（PR，有效维数）
    Sx = np.cov(Xseg.T)
    ev_x = np.linalg.eigvalsh(Sx)
    ev_x = np.maximum(ev_x, 1e-12)
    pr = (ev_x.sum()**2) / (np.sum(ev_x**2))  # 1~2

    # 局部线性动力学拟合 x_{t+1} = A x_t + c
    X_t   = Xseg[:-1]
    X_tp1 = Xseg[1:]
    # 最小二乘估计 A, c
    # 构造 [X_t, 1] 的设计矩阵
    Phi = np.hstack([X_t, np.ones((len(X_t),1))])  # (n-1,3)
    # 单独回归两个维度
    theta1, _, _, _ = lstsq(Phi, X_tp1[:,0], rcond=None)
    theta2, _, _, _ = lstsq(Phi, X_tp1[:,1], rcond=None)
    A = np.vstack([theta1[:2], theta2[:2]])        # 2x2
    c = np.array([theta1[2], theta2[2]])           # 2,
    # 特征值与时间常数
    lam = eigvals(A)  # 2 complex -> 取模
    lam_abs = np.abs(lam)
    tau = -dt / np.log(np.clip(lam_abs, 1e-9, None))  # 实数化的等效时间常数
    # 谱间隙
    lam_sorted = np.sort(lam_abs)
    gap = np.log(lam_sorted[-1]+1e-12) - np.log(lam_sorted[-2]+1e-12)

    # 预测 R^2（简易 3 折）
    def one_step_R2(Xseg):
        X_t   = Xseg[:-1]; X_tp1 = Xseg[1:]
        kf = KFold(n_splits=3, shuffle=False)
        num = 0.0; den = 0.0
        for tr, te in kf.split(X_t):
            Phi_tr = np.hstack([X_t[tr], np.ones((len(tr),1))])
            Phi_te = np.hstack([X_t[te], np.ones((len(te),1))])
            th1, _, _, _ = lstsq(Phi_tr, X_tp1[tr,0], rcond=None)
            th2, _, _, _ = lstsq(Phi_tr, X_tp1[tr,1], rcond=None)
            pred = np.column_stack([Phi_te@th1, Phi_te@th2])
            num += np.sum((X_tp1[te]-pred)**2)
            den += np.sum((X_tp1[te]-X_tp1[te].mean(axis=0))**2)
        return 1 - num/den if den>0 else np.nan
    R2 = one_step_R2(Xseg)

    # 残差噪声各向异性（拟合后的残差协方差特征值比）
    pred_all = (X_t @ A.T) + c
    Res = X_tp1 - pred_all
    Sres = np.cov(Res.T) if len(Res) > 3 else np.eye(2)*np.nan
    ev_res = np.linalg.eigvalsh(Sres) if np.isfinite(Sres).all() else np.array([np.nan, np.nan])
    anisotropy = float(ev_res[-1]/(ev_res[0]+1e-12)) if np.all(np.isfinite(ev_res)) else np.nan

    return dict(
        n=n, dt=dt,
        tortuosity=tortuosity,
        mean_turn=mean_turn,
        vel_evr1=evr1,
        PR=pr,
        lam1=lam_abs.max(), lam2=lam_abs.min(),
        tau1=float(np.sort(tau)[-1]), tau2=float(np.sort(tau)[0]),
        spectral_gap=float(gap),
        R2=float(R2),
        noise_anisotropy=anisotropy
    )

def Local_Field_Flow_matrix(time_axis_ds: np.ndarray,
                           xhat: np.ndarray,
                           seg_df: pd.DataFrame,
                           min_len: int = 12,
                           use_label: str = "auto") -> pd.DataFrame:
    """
    seg_df: DataFrame with columns like:
        - start_idx, end_idx, label, start_time, end_time, label_name
    use_label: "auto" | "label" | "label_name"
        优先使用 'label' (int)，若没有就用 'label_name'
    返回：每个段的一行指标 DataFrame
    """
    # 选择用哪列作为标签
    if use_label == "auto":
        if "label" in seg_df.columns:
            lbl_col = "label"
        elif "label_name" in seg_df.columns:
            lbl_col = "label_name"
        else:
            raise ValueError("seg_df must contain 'label' or 'label_name'.")
    elif use_label in ("label", "label_name"):
        if use_label not in seg_df.columns:
            raise ValueError(f"seg_df does not contain '{use_label}'.")
        lbl_col = use_label
    else:
        raise ValueError("use_label must be 'auto' | 'label' | 'label_name'.")

    need_cols = {"start_idx", "end_idx"}
    if not need_cols.issubset(seg_df.columns):
        raise ValueError("seg_df must contain columns: 'start_idx', 'end_idx'.")

    rows = []
    T = len(time_axis_ds)
    for r in seg_df.itertuples(index=False):
        # 兼容命名：用 getattr 读取
        i0 = int(getattr(r, "start_idx"))
        i1 = int(getattr(r, "end_idx"))
        lbl_val = getattr(r, lbl_col)

        # 越界与长度检查
        i0 = max(0, i0); i1 = min(T, i1)
        if i1 - i0 < min_len:
            continue
        if i0 >= i1:
            continue

        # 取片段
        Xseg = xhat[i0:i1]
        tseg = time_axis_ds[i0:i1]

        m = Local_Field_Flow_Features(Xseg, tseg)
        if m is None:
            continue

        # 标签转 int（如果是 name 就保留一份）
        if lbl_col == "label":
            m["label"] = int(lbl_val)
        else:
            m["label_name"] = str(lbl_val)

        m["start_idx"] = i0
        m["end_idx"] = i1
        rows.append(m)

    return pd.DataFrame(rows)

# =========================================================
# 1) 收集某个行为的所有片段（索引），并拼装回归数据
# =========================================================
def collect_segments(seg_df, behavior, use_label="auto"):
    """
    behavior: 可以是整数标签或字符串名字
    use_label: 'auto' | 'label' | 'label_name'
    返回：列表 [(i0,i1), ...]，按出现顺序
    """
    if use_label == "auto":
        if "label" in seg_df.columns and isinstance(behavior, (int, np.integer)):
            sel = seg_df["label"] == int(behavior)
        elif "label_name" in seg_df.columns and isinstance(behavior, str):
            sel = seg_df["label_name"] == behavior
        else:
            raise ValueError("请指定 use_label='label' 或 'label_name' 与 behavior 类型匹配。")
    elif use_label == "label":
        sel = seg_df["label"] == int(behavior)
    else:
        sel = seg_df["label_name"] == behavior

    segs = [(int(r.start_idx), int(r.end_idx)) for r in seg_df.loc[sel].itertuples(index=False)]
    return segs

def build_regression_data(xhat, time_axis, segs, min_len=5):
    """
    把多段数据拼成一个 (X_t -> X_{t+1}) 的回归数据集（不跨段）
    返回：X_t_all, X_tp1_all, dt_med
    """
    Xts, Xtp1s, dts = [], [], []
    for (i0, i1) in segs:
        if i1 - i0 < min_len: 
            continue
        Xt = xhat[i0:i1-1]
        Xp = xhat[i0+1:i1]
        Xts.append(Xt)
        Xtp1s.append(Xp)
        dts.append(np.median(np.diff(time_axis[i0:i1])))
    if not Xts:
        return None, None, None
    X_t   = np.vstack(Xts)
    X_tp1 = np.vstack(Xtp1s)
    dt_med = float(np.median(dts))
    return X_t, X_tp1, dt_med

# =========================================================
# 2) 拟合离散 LDS：x_{t+1} = A x_t + c
#    并给出连续近似的场：x' = F x + b，F=(A-I)/dt, b=c/dt
# =========================================================
def fit_discrete_lds(X_t, X_tp1, dt, ridge=None):
    """
    ridge: 可设为 float 使用 L2 正则（稳定小样本），否则用最小二乘
    返回：A(2x2), c(2,), F(2x2), b(2,), eigvals(A), misc（字典）
    """
    Phi = np.hstack([X_t, np.ones((len(X_t),1))])  # [x, 1]
    if ridge is None or ridge <= 0:
        # 两个维度分别回归
        theta1, *_ = lstsq(Phi, X_tp1[:,0], rcond=None)
        theta2, *_ = lstsq(Phi, X_tp1[:,1], rcond=None)
    else:
        # 岭回归闭式（(ΦᵀΦ+λI)⁻¹Φᵀy）
        lam = float(ridge)
        G = Phi.T @ Phi + lam*np.eye(Phi.shape[1])
        th1 = np.linalg.solve(G, Phi.T @ X_tp1[:,0])
        th2 = np.linalg.solve(G, Phi.T @ X_tp1[:,1])
        theta1, theta2 = th1, th2

    A = np.vstack([theta1[:2], theta2[:2]])
    c = np.array([theta1[2], theta2[2]])
    # 连续近似（用于画向量场）
    F = (A - np.eye(2)) / dt
    b = c / dt
    lam = eigvals(A)
    return A, c, F, b, lam, dict(theta1=theta1, theta2=theta2)

# =========================================================
# 3) 可视化：轨迹 + 场（quiver）
# =========================================================
def plot_behavior_lds(xhat, time_axis, segs, A, c, F, b, save_dir, title="",
                      grid_res=22, traj_alpha=0.7, cmap_name="viridis",
                      annotate_starts=True, same_limits=None):
    """
    为某个行为画两幅图：左=轨迹叠加（用色条区分第几段），右=动力学向量场
    same_limits: 若传入 (xlim, ylim) 则所有图复用同一坐标范围，便于比较
    """
    # 轨迹范围
    pts = np.vstack([xhat[i0:i1] for (i0,i1) in segs if i1-i0>=2]) if segs else np.empty((0,2))
    if same_limits is None:
        if len(pts) > 0:
            pad = 0.8
            xlim = (pts[:,0].min()-pad, pts[:,0].max()+pad)
            ylim = (pts[:,1].min()-pad, pts[:,1].max()+pad)
        else:
            xlim = (-5,5); ylim=(-5,5)
    else:
        xlim, ylim = same_limits

    # 网格
    xs = np.linspace(xlim[0], xlim[1], grid_res)
    ys = np.linspace(ylim[0], ylim[1], grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    VV = (grid @ F.T) + b      # 连续场速度
    U = VV[:,0].reshape(XX.shape)
    V = VV[:,1].reshape(YY.shape)

    # 画图
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    ax0, ax1 = axes

    # --- 左：轨迹，按段序着色 ---
    cmap = get_cmap(cmap_name, lut=max(1,len(segs)))
    for k, (i0,i1) in enumerate(segs):
        if i1 - i0 < 2: 
            continue
        col = cmap(k/(max(1,len(segs)-1)))
        ax0.plot(xhat[i0:i1,0], xhat[i0:i1,1], lw=1.5, color=col, alpha=traj_alpha)
        # 起点
        if annotate_starts:
            ax0.scatter(xhat[i0,0], xhat[i0,1], s=18, color=col, edgecolor="k", zorder=3)
            ax0.text(xhat[i0,0], xhat[i0,1], f"{k+1}", fontsize=8, color="k",
                     ha="left", va="bottom", alpha=0.9)
    ax0.set_title(f"{title} – trajectories ({len(segs)} segs)")
    ax0.set_xlim(xlim); ax0.set_ylim(ylim)
    ax0.set_xlabel("Latent Dim 1"); ax0.set_ylabel("Latent Dim 2")
    ax0.grid(alpha=0.25)

    # --- 右：动力学流场（quiver） ---
    ax1.quiver(XX, YY, U, V, angles='xy', scale_units='xy', scale=1.0, alpha=0.9)
    # 可叠加所有段轨迹为参考（灰）
    for (i0,i1) in segs:
        if i1 - i0 < 2: 
            continue
        ax1.plot(xhat[i0:i1,0], xhat[i0:i1,1], lw=0.8, color="0.6", alpha=0.5)
    ax1.set_title(f"{title} – vector field")
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)
    ax1.set_xlabel("Latent Dim 1"); ax1.set_ylabel("Latent Dim 2")
    ax1.grid(alpha=0.25)
    
    # --- 平均轨迹 ---
    mean_traj, std_traj, prog = compute_average_trajectory(xhat, segs, n_points=80)
    if mean_traj is not None:
        ax0.plot(mean_traj[:,0], mean_traj[:,1], lw=2, color="red", alpha=0.9, label="mean traj")
        ax1.plot(mean_traj[:,0], mean_traj[:,1], lw=2, color="red", alpha=0.9, label="mean traj")
    
        # 也可画 std 区域 (作为带子)
        ax0.fill_betweenx(mean_traj[:,1],
                          mean_traj[:,0]-std_traj[:,0],
                          mean_traj[:,0]+std_traj[:,0],
                          color="red", alpha=0.1)
    
    plt.tight_layout()
    save_path = f"{save_dir}/{title}.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, axes

# =========================================================
# 4) 一键：对“某一行为”拟合 + 可视化
# =========================================================
def fit_and_plot_behavior_lds(xhat, time_axis, seg_df, behavior, save_dir,
                              use_label="auto", ridge=None,
                              grid_res=22, cmap_name="viridis",
                              share_limits=None):
    """
    behavior: int 或 str
    share_limits: 若提供 (xlim, ylim)，所有行为复用同一坐标范围
    返回：A,c,F,b,lam,(fig,axes)
    """
    segs = collect_segments(seg_df, behavior, use_label=use_label)
    if len(segs) == 0:
        raise ValueError(f"没有找到该行为的片段：{behavior}")

    X_t, X_tp1, dt = build_regression_data(xhat, time_axis, segs, min_len=5)
    if X_t is None:
        raise ValueError("可用片段太短，无法拟合")

    A, c, F, b, lam, _ = fit_discrete_lds(X_t, X_tp1, dt, ridge=ridge)

    # 标题
    beh_name = (behavior if isinstance(behavior, str) else str(behavior))
    title = f"{beh_name} | dt={dt:.3f}s, eig(A)={np.round(np.abs(lam),3)}"

    fig, axes = plot_behavior_lds(
        xhat, time_axis, segs, A, c, F, b, save_dir=save_dir,
        title=title, grid_res=grid_res, cmap_name=cmap_name,
        same_limits=share_limits
    )
    return A, c, F, b, lam, (fig, axes)

# -----------------------------
# 工具：从 seg_df 中按顺序抽取 trial（dm -> tube -> return）
# -----------------------------
from scipy.spatial import ConvexHull
def extract_trials_dm_to_return(seg_df, xhat, kind="win", use_label="label_name"):
    """
    seg_df: DataFrame，至少包含 start_idx, end_idx, 以及 label_name 或 label
    xhat:   (T,2) 潜在态轨迹
    kind:   'win' 或 'go'
    返回：一个列表，每项是 dict：
        {
          'kind': 'win'/'go',
          'dm_row': row_dm(pandas.NamedTuple),
          'tube_row': row_tube,
          'ret_row': row_ret,
          'i0': dm_start_idx,
          'i1': ret_end_idx
        }
    规则：按时间顺序匹配：<kind>_dm -> <kind>_tube -> <kind>_return；缺一则跳过。
    """
    assert {"start_idx", "end_idx"}.issubset(seg_df.columns), "seg_df 需含 start_idx/end_idx"
    df = seg_df.copy().sort_values("start_idx").reset_index(drop=True)

    # 根据列类型选择筛选
    if use_label == "label_name":
        assert "label_name" in df.columns
        names = df["label_name"].astype(str).values
        def is_name(i, nm): return names[i] == nm
    else:
        # 数字 id 的情形；这里假设你会先把 id -> name 的映射灌入一列 'label_name'
        assert "label" in df.columns and "label_name" in df.columns
        names = df["label_name"].astype(str).values
        def is_name(i, nm): return names[i] == nm

    want_dm, want_tube, want_ret = f"{kind}_dm", f"{kind}_tube", f"{kind}_return"
    out = []
    i = 0
    n = len(df)
    while i < n:
        # 找 dm
        if is_name(i, want_dm):
            dm_row = df.iloc[i]
            # 找后续 tube
            j = i + 1
            while j < n and not is_name(j, want_tube):
                j += 1
            if j >= n:
                break
            tube_row = df.iloc[j]
            # 找后续 return
            k = j + 1
            while k < n and not is_name(k, want_ret):
                k += 1
            if k >= n:
                break
            ret_row = df.iloc[k]

            i0 = int(dm_row.start_idx)
            i1 = int(ret_row.end_idx)
            # 防御：范围要合理
            if 0 <= i0 < i1 <= len(xhat):
                out.append(dict(kind=kind, dm_row=dm_row, tube_row=tube_row, ret_row=ret_row, i0=i0, i1=i1))
                # 跳到 return 之后，避免重叠
                i = k + 1
            else:
                i += 1
        else:
            i += 1
    return out

# -----------------------------
# 工具：计算 2D 轨迹的“环面积” = 覆盖所有点的凸包面积
# -----------------------------
def loop_area_convex_hull(xy):
    """
    xy: (N,2) 本 trial 的轨迹点集合（dm_start -> ... -> return_end）
    面积 = 2D 凸包面积（scipy ConvexHull.volume 在 2D 下即为多边形面积）
    轨迹不闭合不影响结果；若点数<3 返回 0
    """
    if xy is None or len(xy) < 3:
        return 0.0
    try:
        hull = ConvexHull(xy)
        return float(hull.volume)  # 2D 中 volume 即为 area
    except Exception:
        return 0.0

# -----------------------------
# 主函数：统计并作图
# -----------------------------
from scipy.stats import ttest_ind
def plot_trial_loop_areas(
    xhat, seg_df,
    save_dir, filename_pdf="trial_loop_areas.pdf",
    csv_name="trial_loop_areas.csv",
    use_label="label_name"
):
    """
    计算并绘制 win / go 两类 trial 的环面积（dm->tube->return）。
    - 每根柱子代表一个 trial 的面积
    - 同时保存 PDF 和 CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取 win / go trials
    trials_win = extract_trials_dm_to_return(seg_df, xhat, kind="win", use_label=use_label)
    trials_go  = extract_trials_dm_to_return(seg_df, xhat, kind="go",  use_label=use_label)

    areas_win = []
    areas_go  = []
    # 保存每个 trial 的详细信息
    rows = []

    # 计算每个 win trial 的面积
    for idx, tr in enumerate(trials_win, start=1):
        pts = xhat[tr["i0"]: tr["i1"]]             # dm 起点到 return 终点的轨迹
        area = loop_area_convex_hull(pts)
        areas_win.append(area)
        rows.append(dict(kind="win", trial_index=idx, start_idx=tr["i0"], end_idx=tr["i1"], area=area))

    # 计算每个 go trial 的面积
    for idx, tr in enumerate(trials_go, start=1):
        pts = xhat[tr["i0"]: tr["i1"]]
        area = loop_area_convex_hull(pts)
        areas_go.append(area)
        rows.append(dict(kind="go", trial_index=idx, start_idx=tr["i0"], end_idx=tr["i1"], area=area))

    # --- 保存 CSV ---
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, csv_name)
    df_out.to_csv(csv_path, index=False)
    print(f"[Saved CSV] {csv_path}")
    return df_out

def plot_trial_area_comparison(
    df_areas, save_dir,
    filename_pdf="trial_area_comparison_norm.pdf",
    csv_name="trial_loop_areas_with_norm.csv"
):
    """
    比较 win/go trial loop areas:
    - 以 go trial 平均面积为基线，做标准化
    - 左侧柱子: go trials，右侧柱子: win trials
    - 柱子 = 均值 ± SEM，叠加原始 trial 点
    - 显著性检验 (unpaired t-test)
    - CSV 保留原始数据和标准化后的数值
    """
    os.makedirs(save_dir, exist_ok=True)

    # 分组
    win_vals = df_areas[df_areas["kind"]=="win"]["area"].dropna().values
    go_vals  = df_areas[df_areas["kind"]=="go"]["area"].dropna().values

    mean_go = np.mean(go_vals)
    if mean_go == 0:
        raise ValueError("Go trials 平均面积为 0，无法标准化")

    # 标准化
    df_areas = df_areas.copy()
    df_areas["area_norm"] = df_areas["area"] / mean_go

    win_norm = df_areas[df_areas["kind"]=="win"]["area_norm"].dropna().values
    go_norm  = df_areas[df_areas["kind"]=="go"]["area_norm"].dropna().values

    # 均值 ± SEM
    mean_win, mean_go_norm = np.mean(win_norm), np.mean(go_norm)
    sem_win  = np.std(win_norm, ddof=1)/np.sqrt(len(win_norm))
    sem_go   = np.std(go_norm, ddof=1)/np.sqrt(len(go_norm))

    # t-test
    tstat, pval = ttest_ind(win_norm, go_norm, equal_var=False)
    print(f"Unpaired t-test (normalized): t={tstat:.3f}, p={pval:.3e}")

    # 绘图
    fig, ax = plt.subplots(figsize=(6,6))
    x = [0,1]  # go 在左，win 在右
    means = [mean_go_norm, mean_win]
    sems  = [sem_go, sem_win]
    labels= ["go","win"]
    colors= ["#3498db","#e74c3c"]

    # 柱子
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.6, capsize=6)

    # 原始数据散点
    jitter = 0.08
    ax.scatter(np.zeros_like(go_norm)+x[0]+np.random.uniform(-jitter,jitter,len(go_norm)),
               go_norm, color=colors[0], alpha=0.7, edgecolor="k", zorder=5)
    ax.scatter(np.zeros_like(win_norm)+x[1]+np.random.uniform(-jitter,jitter,len(win_norm)),
               win_norm, color=colors[1], alpha=0.7, edgecolor="k", zorder=5)

    # 显著性标注
    y_max = max(max(go_norm), max(win_norm)) * 1.2
    ax.plot([x[0],x[1]],[y_max,y_max], color="k", lw=1.2)
    if pval < 0.0001:
        sig = "****"
    elif pval < 0.001:
        sig = "***"
    elif pval < 0.01:
        sig = "**"
    elif pval < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    ax.text(0.5, y_max*1.02, f"{sig}\n(p={pval:.3e})", ha="center", va="bottom", fontsize=10)

    # 样式
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Loop area (normalized, go mean=1)")
    ax.set_title("Normalized trial loop areas")

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, filename_pdf)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved PDF] {pdf_path}")

    # 保存 CSV
    csv_path = os.path.join(save_dir, csv_name)
    df_areas.to_csv(csv_path, index=False)
    print(f"[Saved CSV] {csv_path}")

    return df_areas

# -----------------------------
# 复用：按顺序抽取 trial（win/go 的 dm -> tube -> return）
# -----------------------------
def extract_trials_dm_to_return(seg_df, xhat, kind="win", use_label="label_name"):
    """
    seg_df: DataFrame，包含 start_idx, end_idx, 以及 label_name 或 label
    xhat:   (T,2)
    kind:   'win' 或 'go'
    返回：列表，每项:
        {'kind','dm_row','tube_row','ret_row','i0','i1'}
    """
    assert {"start_idx","end_idx"}.issubset(seg_df.columns)
    df = seg_df.copy().sort_values("start_idx").reset_index(drop=True)

    if use_label == "label_name":
        assert "label_name" in df.columns
        names = df["label_name"].astype(str).values
        def is_name(i, nm): return names[i] == nm
    else:
        assert "label" in df.columns and "label_name" in df.columns
        names = df["label_name"].astype(str).values
        def is_name(i, nm): return names[i] == nm

    want_dm, want_tube, want_ret = f"{kind}_dm", f"{kind}_tube", f"{kind}_return"
    out = []
    i, n = 0, len(df)
    while i < n:
        if is_name(i, want_dm):
            dm_row = df.iloc[i]
            j = i + 1
            while j < n and not is_name(j, want_tube):
                j += 1
            if j >= n: break
            tube_row = df.iloc[j]
            k = j + 1
            while k < n and not is_name(k, want_ret):
                k += 1
            if k >= n: break
            ret_row = df.iloc[k]

            i0 = int(dm_row.start_idx)
            i1 = int(ret_row.end_idx)
            if 0 <= i0 < i1 <= len(xhat):
                out.append(dict(kind=kind, dm_row=dm_row, tube_row=tube_row,
                                ret_row=ret_row, i0=i0, i1=i1))
                i = k + 1
            else:
                i += 1
        else:
            i += 1
    return out

# -----------------------------
# 新增：计算每个 trial 的“环能量”（均方半径）
# -----------------------------
def trial_ring_energy(xy):
    """
    xy: (N,2) dm_start -> ... -> return_end 的轨迹点
    E = mean(||x - x_bar||^2)
    """
    if xy is None or len(xy) < 2:
        return 0.0
    center = xy.mean(axis=0)
    dif = xy - center
    return float(np.mean(np.sum(dif**2, axis=1)))

def compute_ring_energy_per_trial(xhat, seg_df, use_label="label_name"):
    """
    返回 df: kind, trial_index, start_idx, end_idx, energy
    """
    rows = []
    for kind in ["go","win"]:
        trials = extract_trials_dm_to_return(seg_df, xhat, kind=kind, use_label=use_label)
        for idx, tr in enumerate(trials, start=1):
            xy = xhat[tr["i0"]:tr["i1"]]
            E  = trial_ring_energy(xy)
            rows.append(dict(kind=kind, trial_index=idx,
                             start_idx=tr["i0"], end_idx=tr["i1"], energy=E))
    return pd.DataFrame(rows)

# -----------------------------
# 可视化：标准化（go 均值=1）并比较
# -----------------------------
def plot_ring_energy_comparison_norm(
    df_energy, save_dir,
    filename_pdf="trial_ring_energy_comparison_norm.pdf",
    csv_name="trial_ring_energy_with_norm.csv"
):
    """
    - 以 go 的均值为 1 做标准化（energy_norm）
    - 左侧=go，右侧=win；柱子=均值±SEM；叠加 trial 原始点（用标准化后的值）
    - unpaired t-test（基于标准化后），图上标注显著性与 p 值
    - CSV 保存原始 energy 与 energy_norm
    """
    os.makedirs(save_dir, exist_ok=True)

    go_vals  = df_energy[df_energy["kind"]=="go"]["energy"].dropna().values
    win_vals = df_energy[df_energy["kind"]=="win"]["energy"].dropna().values
    if len(go_vals)==0 or len(win_vals)==0:
        raise ValueError("go 或 win 组为空，无法比较。")

    mean_go = np.mean(go_vals)
    if mean_go == 0:
        raise ValueError("go trials 的平均环能量为 0，无法标准化。")

    df_energy = df_energy.copy()
    df_energy["energy_norm"] = df_energy["energy"] / mean_go    # 以 mean go 为基准归一化 得到 win fold change

    go_norm  = df_energy[df_energy["kind"]=="go"]["energy_norm"].values
    win_norm = df_energy[df_energy["kind"]=="win"]["energy_norm"].values

    # 统计量
    mean_go_n, mean_win_n = np.mean(go_norm), np.mean(win_norm)
    sem_go  = np.std(go_norm, ddof=1)/np.sqrt(len(go_norm))
    sem_win = np.std(win_norm, ddof=1)/np.sqrt(len(win_norm))

    # t-test
    tstat, pval = ttest_ind(win_norm, go_norm, equal_var=False)
    print(f"[Ring Energy] Unpaired t-test (normalized): t={tstat:.3f}, p={pval:.3e}")

    # 绘图（go 左、win 右）
    fig, ax = plt.subplots(figsize=(6,6))
    x = [0,1]
    means = [mean_go_n, mean_win_n]
    sems  = [sem_go, sem_win]
    labels= ["go","win"]
    colors= ["#3498db","#e74c3c"]

    # 柱子（均值±SEM）
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.6, capsize=6)

    # 叠加每个 trial 的点（标准化后）
    jitter = 0.08
    ax.scatter(np.zeros_like(go_norm)+x[0]+np.random.uniform(-jitter,jitter,len(go_norm)),
               go_norm, color=colors[0], alpha=0.7, edgecolor="k", zorder=5)
    ax.scatter(np.zeros_like(win_norm)+x[1]+np.random.uniform(-jitter,jitter,len(win_norm)),
               win_norm, color=colors[1], alpha=0.7, edgecolor="k", zorder=5)

    # 显著性标注
    y_max = max(go_norm.max(), win_norm.max()) * 1.2
    ax.plot([x[0],x[1]],[y_max,y_max], color="k", lw=1.2)
    if pval < 1e-4: sig="****"
    elif pval < 1e-3: sig="***"
    elif pval < 1e-2: sig="**"
    elif pval < 5e-2: sig="*"
    else: sig="n.s."
    ax.text(0.5, y_max*1.02, f"{sig}\n(p={pval:.3e})", ha="center", va="bottom", fontsize=10)

    # 样式
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Ring energy (normalized, go mean=1)")
    ax.set_title("Normalized trial ring energy (mean squared radius)")
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, filename_pdf)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved PDF] {pdf_path}")

    # CSV：保留原始值与标准化
    csv_path = os.path.join(save_dir, csv_name)
    df_energy.to_csv(csv_path, index=False)
    print(f"[Saved CSV] {csv_path}")

    return df_energy

# =========================
# 0) 基础：弧长参数化 + 等弧长重采样
# =========================
def _arc_length_path(xy):
    if xy is None or len(xy) < 2:
        s = np.linspace(0, 1, len(xy) if xy is not None else 1)
        return s, 0.0
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1]) if s[-1] > 0 else 1.0
    return s / L, L

def resample_by_arclength(xy, n_points=1024):
    """把一条 (n,2) 轨迹按弧长参数化重采样到固定 n_points（用于求平均轨迹，不改变原始轨迹显示）"""
    if xy is None or len(xy) < 2:
        return None
    s_norm, _ = _arc_length_path(xy)
    f0 = interp1d(s_norm, xy[:, 0], kind="linear", bounds_error=False, fill_value="extrapolate")
    f1 = interp1d(s_norm, xy[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate")
    new_s = np.linspace(0, 1, n_points)
    return np.stack([f0(new_s), f1(new_s)], axis=1)

# =========================
# 1) 从分段表中抽取 trial：<kind>_dm -> <kind>_tube -> <kind>_return
# =========================
def extract_trials(seg_df, xhat, kind="win", use_label="label_name"):
    """返回一个 list，每个元素是该 trial 的 (N,2) 轨迹（dm_start 到 return_end），按时间顺序匹配"""
    df = seg_df.copy().sort_values("start_idx").reset_index(drop=True)
    if use_label == "label_name":
        names = df["label_name"].astype(str).values
        def is_name(i, nm): return names[i] == nm
    else:
        raise ValueError("请使用 use_label='label_name'，并在 seg_df 中提供 label_name 字段。")

    want_dm, want_tube, want_ret = f"{kind}_dm", f"{kind}_tube", f"{kind}_return"
    out = []
    i, n = 0, len(df)
    while i < n:
        if is_name(i, want_dm):
            dm = df.iloc[i]
            j = i + 1
            while j < n and not is_name(j, want_tube):
                j += 1
            if j >= n: break
            tube = df.iloc[j]
            k = j + 1
            while k < n and not is_name(k, want_ret):
                k += 1
            if k >= n: break
            ret = df.iloc[k]
            i0, i1 = int(dm.start_idx), int(ret.end_idx)
            if 0 <= i0 < i1 <= len(xhat):
                out.append(xhat[i0:i1].copy())
                i = k + 1
            else:
                i += 1
        else:
            i += 1
    return out

# =========================
# 2) 组内平均轨迹（按等弧长对齐，整段平均：可保留但不再用于绘制）
# =========================
def mean_trajectory_resampled(traj_list, n_points=1024):
    rs = []
    for xy in traj_list:
        r = resample_by_arclength(xy, n_points=n_points)
        if r is not None:
            rs.append(r)
    if not rs:
        return None, None
    arr = np.stack(rs, axis=0)  # (n_trials, n_points, 2)
    mean_xy = arr.mean(axis=0)
    return mean_xy, arr

# =========================
# 2.1 新增：分段平均（关键点自然对齐）
# =========================
def _mean_keypoint(xhat, seg_df, label_name):
    """取某类片段起点在潜在空间的平均坐标（关键点）"""
    rows = seg_df[seg_df["label_name"] == label_name]
    if rows.empty: return None
    idxs = rows["start_idx"].astype(int).values
    idxs = idxs[(idxs >= 0) & (idxs < len(xhat))]
    if len(idxs) == 0: return None
    return xhat[idxs].mean(axis=0)

def _collect_segments(xhat, seg_df, label_name):
    """收集同一行为（如 win_dm）的所有片段轨迹（start_idx → end_idx）"""
    rows = seg_df[seg_df["label_name"] == label_name].sort_values("start_idx")
    segs = []
    for _, r in rows.iterrows():
        i0, i1 = int(r["start_idx"]), int(r["end_idx"])
        if 0 <= i0 < i1 <= len(xhat):
            segs.append(xhat[i0:i1].copy())
    return segs

def _mean_segment(seg_list, n_points):
    """对一类片段做等弧长重采样并求平均"""
    R = []
    for xy in seg_list:
        rr = resample_by_arclength(xy, n_points=n_points)
        if rr is not None:
            R.append(rr)
    if not R:
        return None, None
    arr = np.stack(R, axis=0)  # (n_seg, n_points, 2)
    mean_xy = arr.mean(axis=0)
    return mean_xy, arr

def average_trial_by_segments(xhat, seg_df, kind="win",
                              n_dm=256, n_tube=512, n_ret=256):
    """
    对某一类（win/go）返回分段平均轨迹与关键点：
    - 分别对 dm / tube / return 求等弧长平均
    - 关键点取各段起点的平均坐标（turn = dm.start，in = tube.start，out = return.start）
    - 绘制时按 dm → tube → return 的顺序画，关键点自然衔接（不强制数值替换）
    """
    dm_lbl   = f"{kind}_dm"
    tube_lbl = f"{kind}_tube"
    ret_lbl  = f"{kind}_return"

    turn_pt = _mean_keypoint(xhat, seg_df, dm_lbl)
    in_pt   = _mean_keypoint(xhat, seg_df, tube_lbl)
    out_pt  = _mean_keypoint(xhat, seg_df, ret_lbl)

    dm_segs   = _collect_segments(xhat, seg_df, dm_lbl)
    tube_segs = _collect_segments(xhat, seg_df, tube_lbl)
    ret_segs  = _collect_segments(xhat, seg_df, ret_lbl)

    dm_mean,   _ = _mean_segment(dm_segs,   n_points=n_dm)
    tube_mean, _ = _mean_segment(tube_segs, n_points=n_tube)
    ret_mean,  _ = _mean_segment(ret_segs,  n_points=n_ret)

    return dict(
        dm_mean=dm_mean, tube_mean=tube_mean, ret_mean=ret_mean,
        turn=turn_pt, _in=in_pt, _out=out_pt
    )

# =========================
# 3) 拟合离散线性动力学 x_{t+1} = A x_t + c （组内整体）
# =========================
def fit_linear_ds(traj_list):
    """
    把一组 trial 连接起来做最小二乘：X_{t+1} = A X_t + c
    返回 A(2x2), c(2,), eigvals(2,), taus(2,)
    """
    X_t_list, X_tp1_list = [], []
    for xy in traj_list:
        if len(xy) < 2: continue
        X_t_list.append(xy[:-1])
        X_tp1_list.append(xy[1:])
    if not X_t_list:
        return None, None, None, None
    X_t = np.vstack(X_t_list)        # (M,2)
    X_tp1 = np.vstack(X_tp1_list)    # (M,2)
    Phi = np.hstack([X_t, np.ones((len(X_t), 1))])  # (M,3)
    th1, _, _, _ = lstsq(Phi, X_tp1[:, 0], rcond=None)
    th2, _, _, _ = lstsq(Phi, X_tp1[:, 1], rcond=None)
    A = np.vstack([th1[:2], th2[:2]])
    c = np.array([th1[2], th2[2]])
    lam = eigvals(A)  # 复数可能
    lam_abs = np.clip(np.abs(lam), 1e-9, None)
    taus = -1.0 / np.log(lam_abs)    # dt=1 的离散等效；若有真实 dt 再乘即可
    return A, c, lam, taus

# =========================
# 4) 速度与角速度（带正负号）
# =========================
def kinematics_metrics(traj_list, time_axis=None):
    """
    返回组内平均速度、平均角速度（保留正负），以及 trial 级别的分布（用于柱形图的误差线）
    - 速度：每步欧式位移 / dt 的平均
    - 角速度：围绕各 trial 的质心，计算角度差分/ dt，取平均（正负号根据逆/顺时针）
    """
    v_trials, w_trials = [], []
    for xy in traj_list:
        if len(xy) < 2:
            continue
        if time_axis is None:
            dt = 1.0
        else:
            dts = np.diff(time_axis)
            dt = float(np.median(dts)) if len(dts) > 0 else 1.0

        dxy = np.diff(xy, axis=0)
        speed = np.linalg.norm(dxy, axis=1) / dt
        v_trials.append(speed.mean())

        center = xy.mean(axis=0)
        xyc = xy - center
        ang = np.arctan2(xyc[:, 1], xyc[:, 0])
        dang = np.unwrap(np.diff(ang))
        omega = dang / dt
        w_trials.append(omega.mean())

    if len(v_trials) == 0:
        return np.nan, np.nan, np.array([]), np.array([])
    return float(np.mean(v_trials)), float(np.mean(w_trials)), np.asarray(v_trials), np.asarray(w_trials)

# =========================
# 5) 绘制向量场（A,c）
# =========================
def plot_vector_field(ax, A, c, bounds, density=24,
                      color="0.5", alpha=0.7,
                      scale=0.16, width=0.004):
    """
    绘制离散线性动力学的方向场 (x_{t+1}-x_t)：
    - scale 越小箭头越长（推荐 0.3~0.8）
    - width 箭头粗细
    - density 网格稠密度
    """
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, density)
    ys = np.linspace(y_min, y_max, density)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    vec = (pts @ A.T) + c - pts
    U = vec[:, 0].reshape(Xg.shape)
    V = vec[:, 1].reshape(Xg.shape)
    ax.quiver(Xg, Yg, U, V, color=color, alpha=alpha,
              angles='xy', scale_units='xy', 
              scale=scale, width=width)

# =========================
# 6) 总流程：提取、均值、可视化 + 参数比较
# =========================
def plot_mean_keypoint(ax, pt, text, marker="o", ms=5, lw=0.8,
                       text_dx=0.06, text_dy=0.06, fontsize=6):
    if pt is None: 
        return
    ax.plot(pt[0], pt[1], marker, color="black", markersize=ms,
            markeredgewidth=lw, zorder=6)
    ax.text(pt[0] + text_dx, pt[1] + text_dy, text, color="black", fontsize=fontsize,
            weight="bold", zorder=6)

def analyze_trials_ring_dynamics(
    xhat, seg_df,
    save_dir, filename_panel="win_go_ring_panel.pdf",
    filename_bars="win_go_dynamics_bars.pdf",
    n_points=1024, time_axis=None
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- 提取 win/go trials ----
    win_trajs = extract_trials(seg_df, xhat, kind="win", use_label="label_name")
    go_trajs  = extract_trials(seg_df, xhat, kind="go",  use_label="label_name")
    assert len(win_trajs)+len(go_trajs) > 0, "没有匹配到任何 trial，请检查分段表。"

    # （可选）整段平均：此后不用于绘制平均线，只保留你需要时使用
    mean_win, win_rs = mean_trajectory_resampled(win_trajs, n_points=n_points)
    mean_go,  go_rs  = mean_trajectory_resampled(go_trajs,  n_points=n_points)

    # ---- 新：分段平均（保证关键点自然对齐：按 dm→tube→return 顺序绘制）----
    mean_go_pack  = average_trial_by_segments(xhat, seg_df, kind="go",  n_dm=512, n_tube=512, n_ret=512)
    mean_win_pack = average_trial_by_segments(xhat, seg_df, kind="win", n_dm=512, n_tube=512, n_ret=512)

    # ---- 拟合整体线性动力学 ----
    A_w, c_w, lam_w, tau_w = fit_linear_ds(win_trajs) if len(win_trajs)>0 else (None,)*4
    A_g, c_g, lam_g, tau_g = fit_linear_ds(go_trajs)  if len(go_trajs)>0  else (None,)*4

    # ---- 速度与角速度（trial 级→组平均）----
    v_w_mean, w_w_mean, v_w_all, w_w_all = kinematics_metrics(win_trajs, time_axis=time_axis)
    v_g_mean, w_g_mean, v_g_all, w_g_all = kinematics_metrics(go_trajs,  time_axis=time_axis)

    # ---- 统一边界 ----
    all_pts = []
    for L in (go_trajs, win_trajs):
        for xy in L:
            all_pts.append(xy)
    XY = np.vstack(all_pts)
    pad = 0.6
    bounds = (XY[:,0].min()-pad, XY[:,0].max()+pad,
              XY[:,1].min()-pad, XY[:,1].max()+pad)
    
    # ---- 面板：1×3（轨迹对比 + GO流场 + WIN流场）----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 子图1：轨迹对比（原始淡轨迹 + 分段平均，按顺序画 dm → tube → return）
    ax = axes[0]
    for xy in go_trajs:
        ax.plot(xy[:,0], xy[:,1], color="#3498db", alpha=0.12, lw=1)
    for xy in win_trajs:
        ax.plot(xy[:,0], xy[:,1], color="#e74c3c", alpha=0.12, lw=1)

    # 依次画 GO 分段平均（单色：蓝）
    if mean_go_pack["dm_mean"]   is not None: ax.plot(mean_go_pack["dm_mean"][:,0],   mean_go_pack["dm_mean"][:,1],   color="#1f77b4", lw=2, label="go mean")
    if mean_go_pack["tube_mean"] is not None: ax.plot(mean_go_pack["tube_mean"][:,0], mean_go_pack["tube_mean"][:,1], color="#1f77b4", lw=2)
    if mean_go_pack["ret_mean"]  is not None: ax.plot(mean_go_pack["ret_mean"][:,0],  mean_go_pack["ret_mean"][:,1],  color="#1f77b4", lw=2)

    # 依次画 WIN 分段平均（单色：红）
    if mean_win_pack["dm_mean"]   is not None: ax.plot(mean_win_pack["dm_mean"][:,0],   mean_win_pack["dm_mean"][:,1],   color="#d62728", lw=2, label="win mean")
    if mean_win_pack["tube_mean"] is not None: ax.plot(mean_win_pack["tube_mean"][:,0], mean_win_pack["tube_mean"][:,1], color="#d62728", lw=2)
    if mean_win_pack["ret_mean"]  is not None: ax.plot(mean_win_pack["ret_mean"][:,0],  mean_win_pack["ret_mean"][:,1],  color="#d62728", lw=2)

    # 平均关键点（只画一个点/标签）
    plot_mean_keypoint(ax, mean_go_pack["turn"], "go turn")
    plot_mean_keypoint(ax, mean_go_pack["_in"],  "go in")
    plot_mean_keypoint(ax, mean_go_pack["_out"], "go out")
    plot_mean_keypoint(ax, mean_win_pack["turn"], "win turn")
    plot_mean_keypoint(ax, mean_win_pack["_in"],  "win in")
    plot_mean_keypoint(ax, mean_win_pack["_out"], "win out")

    ax.set_title("GO vs WIN (segmented means: dm → tube → return)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")

    # 子图2：GO 向量场 + GO 平均（整段平均或分段平均都行；这里用分段平均的整体轮廓）
    ax = axes[1]
    if A_g is not None:
        plot_vector_field(ax, A_g, c_g, bounds=bounds,
                          density=24, color="#1f77b4", alpha=0.5,
                          scale=0.16, width=0.004)
    # 叠加 go 分段平均的轮廓（顺序画）
    if mean_go_pack["dm_mean"]   is not None: ax.plot(mean_go_pack["dm_mean"][:,0],   mean_go_pack["dm_mean"][:,1],   color="#1f77b4", lw=2.6)
    if mean_go_pack["tube_mean"] is not None: ax.plot(mean_go_pack["tube_mean"][:,0], mean_go_pack["tube_mean"][:,1], color="#1f77b4", lw=2.6)
    if mean_go_pack["ret_mean"]  is not None: ax.plot(mean_go_pack["ret_mean"][:,0],  mean_go_pack["ret_mean"][:,1],  color="#1f77b4", lw=2.6)
    plot_mean_keypoint(ax, mean_go_pack["turn"], "go turn")
    plot_mean_keypoint(ax, mean_go_pack["_in"],  "go in")
    plot_mean_keypoint(ax, mean_go_pack["_out"], "go out")
    ax.set_title("GO: fitted vector field + segmented mean")
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.25)

    # 子图3：WIN 向量场 + WIN 平均
    ax = axes[2]
    if A_w is not None:
        plot_vector_field(ax, A_w, c_w, bounds=bounds,
                          density=24, color="#d62728", alpha=0.5,
                          scale=0.16, width=0.004)
    if mean_win_pack["dm_mean"]   is not None: ax.plot(mean_win_pack["dm_mean"][:,0],   mean_win_pack["dm_mean"][:,1],   color="#d62728", lw=2.6)
    if mean_win_pack["tube_mean"] is not None: ax.plot(mean_win_pack["tube_mean"][:,0], mean_win_pack["tube_mean"][:,1], color="#d62728", lw=2.6)
    if mean_win_pack["ret_mean"]  is not None: ax.plot(mean_win_pack["ret_mean"][:,0],  mean_win_pack["ret_mean"][:,1],  color="#d62728", lw=2.6)
    plot_mean_keypoint(ax, mean_win_pack["turn"], "win turn")
    plot_mean_keypoint(ax, mean_win_pack["_in"],  "win in")
    plot_mean_keypoint(ax, mean_win_pack["_out"], "win out")
    ax.set_title("WIN: fitted vector field + segmented mean")
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.25)

    plt.tight_layout()
    panel_path = os.path.join(save_dir, filename_panel)
    plt.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {panel_path}")

    # ---- 参数比较：两个时间常数 + 平均速度 + 平均角速度（带正负） ----
    def taus_from(lam):
        if lam is None: return [np.nan, np.nan]
        lam_abs = np.clip(np.abs(lam), 1e-9, None)
        taus = -1.0 / np.log(lam_abs)
        taus = np.sort(np.real(taus))  # 小的快、大的慢；展示实部
        return [taus[0], taus[-1]]

    tau_g_pair = taus_from(lam_g)
    tau_w_pair = taus_from(lam_w)

    def mean_sem(arr):
        if arr is None or len(arr)==0: return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr, ddof=1)/np.sqrt(len(arr)))

    v_g_m, v_g_se = mean_sem(v_g_all)
    v_w_m, v_w_se = mean_sem(v_w_all)
    w_g_m, w_g_se = mean_sem(w_g_all)
    w_w_m, w_w_se = mean_sem(w_w_all)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) 两个时间常数（整体拟合；如需误差可做 bootstrap）
    ax = axes[0]
    width = 0.35
    x = np.arange(2)  # tau_fast, tau_slow
    ax.bar(x - width/2, [tau_g_pair[0], tau_g_pair[1]], width, color="#1f77b4", alpha=0.85, label="go")
    ax.bar(x + width/2, [tau_w_pair[0], tau_w_pair[1]], width, color="#d62728", alpha=0.85, label="win")
    ax.set_xticks(x); ax.set_xticklabels(["tau_fast", "tau_slow"])
    ax.set_ylabel("time constant (discrete)")
    ax.set_title("Time constants from A")
    ax.legend()

    # (b) 平均速度（trial 分布 → 均值±SEM）
    ax = axes[1]
    ax.bar([0 - width/2, 0 + width/2], [v_g_m, v_w_m], width,
           yerr=[v_g_se, v_w_se], capsize=5,
           color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.set_xticks([0 - width/2, 0 + width/2]); ax.set_xticklabels(["go", "win"])
    ax.set_ylabel("mean speed")
    ax.set_title("Mean speed (trial-level)")

    # (c) 平均角速度（trial 分布 → 均值±SEM，保留正负）
    ax = axes[2]
    ax.bar([0 - width/2, 0 + width/2], [w_g_m, w_w_m], width,
           yerr=[w_g_se, w_w_se], capsize=5,
           color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.set_xticks([0 - width/2, 0 + width/2]); ax.set_xticklabels(["go", "win"])
    ax.set_ylabel("mean angular velocity")
    ax.set_title("Mean angular velocity (signed)")

    plt.tight_layout()
    bars_path = os.path.join(save_dir, filename_bars)
    plt.savefig(bars_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {bars_path}")

    return dict(
        mean_win_segments=mean_win_pack, mean_go_segments=mean_go_pack,
        A_win=A_w, c_win=c_w, eig_win=lam_w, tau_win=tau_w,
        A_go=A_g,  c_go=c_g,  eig_go=lam_g,  tau_go=tau_g,
        v_win_all=v_w_all, w_win_all=w_w_all,
        v_go_all=v_g_all,   w_go_all=w_g_all
    )

## posterior vs looparea
from scipy.stats import pearsonr, spearmanr
def analyze_looparea_posterior_correlation(
    df_areas,
    posterior_csv,
    save_dir,
    filename_scatter="looparea_vs_posterior_scatter.pdf",
    filename_corr="looparea_posterior_corr.pdf",
    csv_name="looparea_with_posterior.csv"
):
    """
    将 trial loop area 与 HMM posterior probability 对齐，做相关性分析
    - df_areas: 来自 plot_trial_loop_areas 的 DataFrame
    - posterior_csv: posterior probability 的 CSV 文件路径
        列名: "0"=win-like, "1"=random-like, "2"=go-like
        行: trial (按时间顺序)
    - 输出:
        - 合并数据 CSV
        - 散点相关图 (loop area vs posterior)
        - 各状态相关系数柱形图
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 读取 posterior ---
    df_post = pd.read_csv(posterior_csv)
    assert "0" in df_post.columns and "1" in df_post.columns and "2" in df_post.columns, \
        "posterior_csv 缺少列名 '0','1','2'"

    # --- 按 trial 对齐 ---
    df_areas = df_areas = df_areas.sort_values("start_idx").reset_index(drop=True)
    df_post = df_post.copy().reset_index(drop=True)
    n = min(len(df_areas), len(df_post))
    df_areas = df_areas.iloc[:n, :].reset_index(drop=True)
    df_post = df_post.iloc[:n, :].reset_index(drop=True)

    # 合并
    df_combined = pd.concat([df_areas, df_post], axis=1)
    csv_path = os.path.join(save_dir, csv_name)
    df_combined.to_csv(csv_path, index=False)
    print(f"[Saved CSV] {csv_path}")

    # --- 相关性计算 ---
    corrs = []
    for col in ["0","1","2"]:
        pear_r, pear_p = pearsonr(df_combined["area"], df_combined[col])
        spear_r, spear_p = spearmanr(df_combined["area"], df_combined[col])
        corrs.append(dict(
            state=col,
            pearson_r=pear_r, pearson_p=pear_p,
            spearman_r=spear_r, spearman_p=spear_p
        ))
    df_corr = pd.DataFrame(corrs)

    # --- 散点图 (每个 posterior vs area) ---
    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
    state_names = {"0":"win-like", "1":"random-like", "2":"go-like"}
    colors = {"0":"#e74c3c", "1":"#95a5a6", "2":"#3498db"}
    for i, col in enumerate(["0","1","2"]):
        ax = axes[i]
        sns.regplot(x=col, y="area", data=df_combined,
                    scatter_kws=dict(color=colors[col], alpha=0.6),
                    line_kws=dict(color="k", lw=1.2),
                    ax=ax)
        ax.set_title(f"Loop area vs {state_names[col]}\n"
                     f"Pearson r={df_corr.loc[i,'pearson_r']:.2f}, p={df_corr.loc[i,'pearson_p']:.3e}")
        ax.set_xlabel(f"Posterior {state_names[col]}")
        if i==0: ax.set_ylabel("Loop area")
    plt.tight_layout()
    pdf_path = os.path.join(save_dir, filename_scatter)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved scatter PDF] {pdf_path}")

    # --- 柱形图总结相关系数 ---
    fig, ax = plt.subplots(figsize=(6,6))
    sns.barplot(x="state", y="pearson_r", data=df_corr,
                palette=[colors[c] for c in df_corr["state"]], ax=ax)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticklabels([state_names[s] for s in df_corr["state"]])
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_title("Correlation between loop area and posterior")
    for i,row in df_corr.iterrows():
        ax.text(i, row["pearson_r"] + 0.02*np.sign(row["pearson_r"]),
                f"p={row['pearson_p']:.3e}", ha="center", fontsize=9)
    plt.tight_layout()
    pdf_path = os.path.join(save_dir, filename_corr)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved corr PDF] {pdf_path}")

    return df_combined, df_corr